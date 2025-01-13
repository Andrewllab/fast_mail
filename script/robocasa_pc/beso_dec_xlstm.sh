python run_test.py  --config-name=robocasa_horeka_pc_config \
            --multirun agents=beso_agent \
            agent_name=beso_xlstm \
            group=beso_decoder_only_benchmark \
            agents/model=beso/beso_dec_xlstm \
            xlstm_encoder_blocks=14 \
            seed=0