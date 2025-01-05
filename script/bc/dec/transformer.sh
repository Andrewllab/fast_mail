python run.py  --config-name=libero_horeka_config \
            --multirun agents=bc_agent \
            agent_name=bc_transformer \
            group=bc_decoder_only_non_ema \
            agents/model=bc/bc_dec_transformer \
            task_suite=libero_spatial \
            if_use_ema=False \
            seed=0,1,2