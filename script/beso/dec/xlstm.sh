python run.py  --config-name=libero_horeka_config \
            --multirun agents=beso_agent \
            agent_name=beso_xlstm \
            group=beso_decoder_only_test \
            agents/model=beso/beso_dec_xlstm \
            task_suite=libero_spatial \
            traj_per_task=10 \
            seed=0,1,2