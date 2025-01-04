python run.py  --config-name=libero_horeka_config \
            --multirun agents=bc_agent \
            agent_name=bc_transformer \
            group=bc_decoder_only \
            agents/model=bc/bc_dec_transformer \
            task_suite=libero_object,libero_goal,libero_10,libero_spatial \
            seed=0,1,2