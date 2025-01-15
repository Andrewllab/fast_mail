export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
export CC=/opt/gcc/11/bin/gcc
export CXX=/opt/gcc/11/bin/g++
export CUB_HOME=$HOME/project/atalay_master_thesis/3D-Diffusion-Policy/third_party/cub-2.1.0
export CXXFLAGS="-O2 -march=core-avx2"
export CFLAGS="-O2 -march=core-avx2"

python run_test.py  --config-name=robocasa_horeka_pc_config \
            --multirun agents=beso_agent \
            agent_name=beso_xlstm \
            group=beso_decoder_only_benchmark \
            agents/model=beso/beso_dec_xlstm \
            agents/obs_encoders=point_mlp \
            xlstm_encoder_blocks=8 \
            scaler_type='minmax' \
            seed=0