python run_test.py  --config-name=robocasa_horeka_config \
            --multirun agents=beso_agent \
            agent_name=beso_mamba \
            group=beso_encoder_decoder_benchmark \
            agents/model=beso/beso_encdec_mamba \
            mamba_n_layer_encoder=4 \
            mamba_n_layer_decoder=8 \
            seed=0,1,2