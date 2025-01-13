python run_test.py  --config-name=robocasa_horeka_pc_config \
            --multirun agents=beso_agent \
            agent_name=beso_mamba \
            group=beso_decoder_only_benchmark \
            agents/model=beso/beso_dec_mamba \
            mamba_n_layer_encoder=10 \
            seed=0