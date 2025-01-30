export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
export CC=/opt/gcc/11/bin/gcc
export CXX=/opt/gcc/11/bin/g++
export CUB_HOME=$HOME/project/atalay_master_thesis/3D-Diffusion-Policy/third_party/cub-2.1.0
export CXXFLAGS="-O2 -march=core-avx2"
export CFLAGS="-O2 -march=core-avx2"

python run_test.py  --config-name=robocasa_horeka_pc_config \
            --multirun agents=beso_agent \
            agent_name=beso_transformer \
            group=beso_decoder_only_max \
            agents/model=beso/beso_dec_transformer \
            agents/obs_encoders=point_mlp_pooling \
            encoder_n_layer=6 \
            use_pos_emb=True \
            n_embd=256,512 \
            latent_dim=128,256 \
            seed=0,1,2