"""
Neural network model used for DQN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Neural network for Q-function approximation in DQN with integrated action masking.
    
    This architecture uses:
    1. Convolutional layers to capture spatial patterns in the grid
    2. Fully connected layers to map convolutional features to Q-values
    3. Integrated action masking to set -inf values to cells not on the firefront
    """
    def __init__(self, width=10, height=10, channels=5):
        super(QNetwork, self).__init__()
        
        # Import config here to avoid circular imports
        from src.rl.config import NETWORK
        
        # Store grid dimensions
        self.width = width
        self.height = height
        
        num_actions = width * height
        
        # Get network architecture from config

        conv_filters = NETWORK["conv_filters"]
        conv_kernel_size = NETWORK["conv_kernel_size"]
        use_fc_layers = NETWORK["use_fc_layers"]
        hidden_fc_config = NETWORK["hidden_fc"]

        # Allow kernel size and padding to be specified as int or list
        if isinstance(conv_kernel_size, int):
            conv_kernel_sizes = [conv_kernel_size] * len(conv_filters)
        else:
            conv_kernel_sizes = conv_kernel_size
        conv_paddings = [k // 2 for k in conv_kernel_sizes]  # Compute padding from kernel sizes

        # Parse hidden_fc depending on whether it is an int (one layer) or a list (multiple layers)
        if isinstance(hidden_fc_config, int):
            hidden_fc_sizes = [hidden_fc_config]
        elif isinstance(hidden_fc_config, list):
            hidden_fc_sizes = hidden_fc_config
        else:
            raise ValueError(f"hidden_fc must be int or list, got {type(hidden_fc_config)}")
        

        # Build convolutional layers with per-layer kernel size and padding
        conv_layers = []
        in_channels = channels
        self.input_pad = conv_paddings[0] if len(conv_paddings) > 0 else 0
        for i, filters in enumerate(conv_filters):
            ksize = conv_kernel_sizes[i]
            pad = conv_paddings[i]
            # Only pad input for first layer, as before
            if i == 0:
                conv_layers.append(nn.Conv2d(in_channels, filters, kernel_size=ksize, padding=0))
            else:
                conv_layers.append(nn.Conv2d(in_channels, filters, kernel_size=ksize, padding=pad))
            conv_layers.append(nn.ReLU())
            in_channels = filters
        self.conv_network = nn.Sequential(*conv_layers)

        # Compute output size for FC layers
        conv_output_size = in_channels * width * height
        
        # Build output layers based on use_fc_layers
        if use_fc_layers and hidden_fc_sizes:
            # Use fully connected layers
            fc_layers = []
            current_size = conv_output_size
            
            for hidden_size in hidden_fc_sizes:
                fc_layers.extend([
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU()
                ])
                current_size = hidden_size
            
            fc_layers.append(nn.Linear(current_size, num_actions))
            self.fc_network = nn.Sequential(*fc_layers)
            self.output_conv = None
        else:
            # Preserve spatial structure with 1x1 convolution
            self.fc_network = None
            # Map from last conv channels to 1 channel (Q-values per spatial location)
            self.output_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        # Register fixed (non-trainable) kernel for identifying alive neighbors in action mask computation
        neighbor_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        neighbor_kernel[0, 0, 1, 1] = 0  # Set center to 0 to exclude the cell itself
        self.register_buffer('neighbor_kernel', neighbor_kernel)
        
        # Pre-cache tensor constants
        self.register_buffer('mask_zero', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('mask_neg_inf', torch.tensor(-torch.inf, dtype=torch.float32))
        
    def _compute_action_mask(self, x):
        """
        Compute action mask directly from the state tensor (x) using efficient tensor operations.
        """
        batch_size = x.size(0)
        
        # Extract status channels: ALIVE (channel 2), BURNING (channel 3), BURNT (channel 4)
        alive_channel = x[:, 2, :, :]
        burning_channel = x[:, 3, :, :]
        
        # Check which cells are burning
        is_burning = burning_channel > 0.5
        
        # Count alive neighbors in each 3x3 neighborhood
        alive_neighbor_count = F.conv2d(
            alive_channel.unsqueeze(1),
            self.neighbor_kernel,
            padding=1
        ).squeeze(1)
        
        # Check if any neighbor is alive
        has_alive_neighbor = alive_neighbor_count > 0
        
        # Valid action if burning AND has alive neighbor and if none exists then all burning cells are valid
        valid_actions = is_burning & has_alive_neighbor
        has_any_valid = valid_actions.view(batch_size, -1).any(dim=1, keepdim=True)
        has_any_valid = has_any_valid.view(batch_size, 1, 1)
        valid_actions = torch.where(has_any_valid, valid_actions, is_burning)
        valid_actions_flat = valid_actions.view(batch_size, -1)
        
        # Create action mask: 0 for valid actions, -inf for invalid
        action_mask = torch.where(
            valid_actions_flat,
            self.mask_zero,
            self.mask_neg_inf
        )
        
        return action_mask
    
    def forward(self, x):
        """
        Forward pass through the network with integrated action masking.
        
        Args:
            x: Input tensor representing the state [batch_size, width, height, channels]
            
        Returns:
            Q-values for each action with integrated masking
        """
        batch_size = x.size(0)
        
        # Copy input to avoid modifying original tensor
        x = x.clone()
        
        # Reshape input for convolutional layers [batch_size, channels, width, height]
        x = x.permute(0, 3, 1, 2)

        # Compute action mask from the input state
        action_mask = self._compute_action_mask(x)

        # Pass through all convolutional layers
        # Pad the input only (not intermediate activations) with 1s, using first layer's padding
        if self.input_pad > 0:
            x = F.pad(x, (self.input_pad, self.input_pad, self.input_pad, self.input_pad), mode='constant', value=1.0)
        x = self.conv_network(x)

        # Generate Q-values based on architecture choice
        if self.fc_network is not None:
            # Use fully connected layers - flatten the conv output
            x_flattened = x.reshape(batch_size, -1)
            q_values = self.fc_network(x_flattened)
        else:
            # Preserve spatial structure - use 1x1 conv and flatten at the end
            x = self.output_conv(x)  # [batch_size, 1, width, height]
            q_values = x.reshape(batch_size, -1)  # [batch_size, width*height]
        
        # Apply negative activation to ensure Q-values are non-positive, since we know that all rewards are non-positive
        q_values = -F.softplus(q_values)

        # Apply action mask by adding it to Q-values
        # Valid actions have mask=0 (no change), invalid actions have mask=-inf
        masked_q_values = q_values + action_mask
        
        return masked_q_values