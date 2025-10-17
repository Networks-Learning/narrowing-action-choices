# Human subject study dataset
The human subject study data are in `study_data.csv`. The data include the following columns:

* **participant_id (str)**: anonymous participant identifier.
* **study_id (str)**:  identifier of the prolific study.
* **session_id (str)**: study session identifier.
* **trial_batch_id (int)**: number of the batch of the game instances assigned to a participant.
* **gameNum (int)**: number the game instance.
* **mapId (int)**: identifier of the forest map, takes values in {0, 1, .., 9}.
* **initLocId (int)**: identifier of the fire spread at the beginning of the game, takes values in {0,1,2,3}.
* **seed (int)**: the random seed controlling the fire spread dynamics and noise value for the action sets.
* **epsilon (float)**: the $\varepsilon$ value, takes values in [0.0, 1.0].
* **start_time (int)**: timestamp of the game beginning (ms).
* **time_step (int)**: time step of the game.
* **action_x (int)**: x coordinate of the grid tile clicked by the participant, takes values in {0, 1, .., 9}.
* **action_y (int)**: y coordinate of the grid tile clicked by the participant, takes values in {0, 1, .., 9}.
* **reward (int)**: minus the number of the tiles that caught fire after an action.
* **score (int)**: number of healthy tiles after taking an action.
* **time (int)** timestamp of the action taken (similar to start time).







