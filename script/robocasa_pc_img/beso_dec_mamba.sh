export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
export CC=/opt/gcc/11/bin/gcc
export CXX=/opt/gcc/11/bin/g++
export CUB_HOME=$HOME/project/atalay_master_thesis/3D-Diffusion-Policy/third_party/cub-2.1.0
export CXXFLAGS="-O2 -march=core-avx2"
export CFLAGS="-O2 -march=core-avx2"

python run_test.py  --config-name=robocasa_horeka_pc_img_config \
            --multirun agents=beso_agent \
            agent_name=beso_mamba \
            group=beso_decoder_only_new \
            agents/model=beso/beso_dec_mamba \
            agents/obs_encoders=point_img_encoder \
            agents.if_film_condition=True \
            mamba_n_layer_encoder=8 \
            use_pos_emb=True \
            seed=0,1,2