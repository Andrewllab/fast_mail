python run.py  --config-name=libero_horeka_config \
            --multirun agents=vqbet_agent \
            agent_name=vqbet \
            group=vqbet_decoder_only \
            agents/model=vqbet/vqbet_dec_transformer \
            task_suite=libero_object \
            seed=0,1,2