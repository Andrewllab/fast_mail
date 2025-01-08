python run_test.py  --config-name=robocasa_horeka_config \
            --multirun agents=beso_agent \
            agent_name=beso_transformer \
            group=beso_decoder_only \
            agents/model=beso/beso_dec_transformer \
            encoder_n_layer=6 \
            seed=0