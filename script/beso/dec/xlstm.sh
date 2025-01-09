python run.py  --config-name=libero_horeka_config \
            --multirun agents=beso_agent \
            agent_name=beso_xlstm \
            group=beso_decoder_only \
            agents/model=beso/beso_dec_xlstm \
            task_suite=libero_object,libero_goal,libero_10,libero_spatial \
            traj_per_task=10,50 \
            seed=0,1,2