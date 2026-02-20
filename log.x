HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train platform=horeka data=robocasa_coffee_setup_mug obs_modality=robocasa_track_one_tracked_one_not obs_encoder=pakt_encoder experiment=none agent.noise_model.decoder.n_layers=6,8 data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False seed=42,420,1337 +hydra.launcher.additional_parameters.dependency="afterany:3861235" --multirun

HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train platform=horeka data=robocasa_doors_close_double obs_modality=robocasa_track obs_encoder=pakt_encoder experiment=none agent.noise_model.decoder.n_layers=6 +data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True --multirun

wandb.project=pakt_robocasa_pnp_counter_to_stove
trainer.max_epochs=200
decoder.n_layers=8
optimizer.lr=0.0001
noise_distribution.scale=0.5
action_obs_tokenizer.embed_dim=512
t02_to_pakt_action_obs.include_tracked_in_action=True



HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train platform=horeka data=robocasa_doors_open_single obs_modality=robocasa_track_one_tracked obs_encoder=pakt_encoder experiment=none agent.noise_model.decoder.n_layers=6 data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True --multirun



HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train platform=horeka data=robocasa_coffee_setup_mug,robocasa_pnp_cab_to_counter,robocasa_pnp_counter_to_cab,robocasa_pnp_counter_to_microwave,robocasa_pnp_counter_to_sink,robocasa_pnp_counter_to_stove,robocasa_pnp_microwave_to_counter,robocasa_pnp_sink_to_counter,robocasa_pnp_stove_to_counter,robocasa_coffee_serve_mug obs_modality=robocasa_track_one_tracked_one_not obs_encoder=pakt_encoder experiment=none agent.noise_model.decoder.n_layers=6 data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True --multirun



# One tracked one not
robocasa_coffee_setup_mug
robocasa_pnp_cab_to_counter
robocasa_pnp_counter_to_cab
robocasa_pnp_counter_to_microwave
robocasa_pnp_counter_to_sink
robocasa_pnp_counter_to_stove
robocasa_pnp_microwave_to_counter
robocasa_pnp_sink_to_counter
robocasa_pnp_stove_to_counter
robocasa_coffee_serve_mug

robocasa_coffee_setup_mug,robocasa_pnp_cab_to_counter,robocasa_pnp_counter_to_cab,robocasa_pnp_counter_to_microwave,robocasa_pnp_counter_to_sink,robocasa_pnp_counter_to_stove,robocasa_pnp_microwave_to_counter,robocasa_pnp_sink_to_counter,robocasa_pnp_stove_to_counter,robocasa_coffee_serve_mug


# One tracked
robocasa_doors_close_double 0 x
robocasa_doors_close_single 1 - (file not found)
robocasa_doors_open_double 2 - (file not found)
robocasa_doors_open_single 3 - not even wandb created

robocasa_drawer_close
robocasa_drawer_open

robocasa_microwave_turn_off
robocasa_microwave_turn_on

robocasa_sink_turn_off_faucet
robocasa_sink_turn_on_faucet
robocasa_sink_turn_spout

robocasa_coffee_press_button


robocasa_doors_close_double,robocasa_doors_close_single,robocasa_doors_open_double,robocasa_doors_open_single,robocasa_drawer_close,robocasa_drawer_open,robocasa_microwave_turn_off,robocasa_microwave_turn_on,robocasa_sink_turn_off_faucet,robocasa_sink_turn_on_faucet,robocasa_sink_turn_spout,robocasa_coffee_press_button

# Needs goal embedding
robocasa_stove_turn_off
robocasa_stove_turn_on

# Forgot to add something




# Main reference tasks for grid search:

robocasa_doors_open_single
robocasa_pnp_counter_to_stove
robocasa_microwave_turn_on
robocasa_stove_turn_on

Modalities:
robocasa_track_one_tracked
robocasa_track_one_tracked_one_not

Options:
goals=clip_embed

Hyperparameters grid_search 1:
  agent.noise_model.decoder.n_layers=6,8\
  agent.noise_model.decoder.attention.n_heads=4,8\
  agent.noise_model.decoder.residual_dropout=0.05,0.1\
  agent.noise_model.dropout_prob=0.05,0.1\
  data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False\

Hyperparameters grid_search 2:
  data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False\
  data.gpu_batch_transforms.t01_fps_prep.n_points=20,40,80\
  agent.noise_model.action_obs_tokenizer.pos_encoder.hidden_dim=128,256\
  agent.optimizer.lr=3e-5,1e-4,3e-4\

Hyperparameters grid_search 3:
  data.gpu_batch_transforms.t01_fps_prep.n_points=20,40,60\
  agent.optimizer.weight_decay=0.0,0.1\
  agent.lr_scheduler.warmup_ratio=0.03,0.1\
  data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False\

HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train\
  experiment=none\
  platform=horeka\
  data=robocasa_stove_turn_on\
  obs_modality=robocasa_track_one_tracked\
  obs_encoder=pakt_encoder\
  data.gpu_batch_transforms.t01_fps_prep.n_points=20,40,60\
  agent.optimizer.weight_decay=0.0,0.1\
  agent.lr_scheduler.warmup_ratio=0.03,0.1\
  data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False\
  hydra.launcher.name="pakt_grid_sweep_microwave"\
  logger.wandb.project="PAKT_grid_search"\
  logger.wandb.tags="[grid_search_03]"\
  goals=clip_embed\
  --multirun


HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train  experiment=none  platform=horeka  data=robocasa_doors_open_single  obs_modality=robocasa_track_one_tracked  obs_encoder=pakt_encoder  data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False  data.gpu_batch_transforms.t01_fps_prep.n_points=20,40,80  agent.noise_model.action_obs_tokenizer.pos_encoder.hidden_dim=128,256  agent.optimizer.lr=3e-5,1e-4,3e-4  hydra.launcher.name="pakt_grid_sweep_stove_on"  logger.wandb.project="PAKT_grid_search"  logger.wandb.tags="[grid_search_02]"  --multirun &
HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train  experiment=none  platform=horeka  data=robocasa_pnp_counter_to_stove  obs_modality=robocasa_track_one_tracked_one_not  obs_encoder=pakt_encoder  data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False  data.gpu_batch_transforms.t01_fps_prep.n_points=20,40,80  agent.noise_model.action_obs_tokenizer.pos_encoder.hidden_dim=128,256  agent.optimizer.lr=3e-5,1e-4,3e-4  hydra.launcher.name="pakt_grid_sweep_stove_on"  logger.wandb.project="PAKT_grid_search"  logger.wandb.tags="[grid_search_02]"  --multirun &
HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train  experiment=none  platform=horeka  data=robocasa_microwave_turn_on  obs_modality=robocasa_track_one_tracked  obs_encoder=pakt_encoder  data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False  data.gpu_batch_transforms.t01_fps_prep.n_points=20,40,80  agent.noise_model.action_obs_tokenizer.pos_encoder.hidden_dim=128,256  agent.optimizer.lr=3e-5,1e-4,3e-4  hydra.launcher.name="pakt_grid_sweep_stove_on"  logger.wandb.project="PAKT_grid_search"  logger.wandb.tags="[grid_search_02]"  --multirun &
HYDRA_FULL_ERROR=1 python train.py --config-name pakt_train  experiment=none  platform=horeka  data=robocasa_stove_turn_on  obs_modality=robocasa_track_one_tracked  obs_encoder=pakt_encoder  data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action=True,False  data.gpu_batch_transforms.t01_fps_prep.n_points=20,40,80  agent.noise_model.action_obs_tokenizer.pos_encoder.hidden_dim=128,256  agent.optimizer.lr=3e-5,1e-4,3e-4  hydra.launcher.name="pakt_grid_sweep_stove_on"  logger.wandb.project="PAKT_grid_search"  logger.wandb.tags="[grid_search_02]"  goals=clip_embed  --multirun &
