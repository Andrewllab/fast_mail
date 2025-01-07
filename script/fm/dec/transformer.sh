python run.py  --config-name=libero_horeka_config \
            --multirun agents=fm_agent \
            agent_name=fm_transformer \
            group=fm_decoder_only \
            agents/model=fm/fm_dec_transformer \
            task_suite=libero_object \
            seed=0,1,2