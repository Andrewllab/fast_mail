import wandb

api = wandb.Api()
train_runs = api.runs(
    "nicolasschreiber/PAKT_grid_search",
    filters={"$and": [{"state": "finished"}, {"jobType": "train"}]},
)
test_runs = api.runs(
    "nicolasschreiber/PAKT_grid_search",
    filters={
        "$and": [
            {"$or": [{"state": "finished"}, {"state": "running"}]},
            {"jobType": "test_sim"},
        ]
    },
)

train_run_ids = []

prev_ids = "ae5rc6ra,pmh0sf2r,5pru6pi2,oceflxtk,zk8qnugk,3hz5f599,mhg7xdfb,nesd94mq,ohxdtlbg,czwcu2ub,zi5e032l,azz4ys47,yz5r90c7,p8a6li82,3oan0gbn,aklypnx9,ijja6wew,hjm1fiea,qnaync8f,wodly66s,mceb15m4,xog8i92n,fbg4un5w,tlbjzs5e,ai1c775o,gd8coftk,h9ut7327,0yggfbku,efl1nkc4,rnni2zhh,vxzuumex,f7mexygk,i39jzf6g,tc5vwosy,qdbzh6k5,8ltbwpqc,ndmkpbvq,a6p9cyri,b7k3sqev,elopl1wq,ja51t6a4,0d56rgmm,0xk3gaoj,tgo6qacz,1czthxjw,84q3h6e2,af1eaig5,zt894je7,lzr63awa,c7qdleji,5g18smkv,eut6clpq,d2n0x3fe,sy0gf9p6,u5lsbn2w,wd091m6g,k0ygtcfw,nqxl80db,amh4yycv,r8xdnj9x,zkp5zgs7,99dxjvgz,cu54ck7p,1vcd49f0,luetusde,77b2rqsr,gh8s6c3u,7geigmsc,l97hyt2b,cz6ojxw2,g5q8w8pk,mm2klk0l,qxzjh6iw,3ee1falx,vharnh56,7chv8q0l,re4v8pw9,zp55m35o,a9e3me8t,iweq3pb7,imw75ksj,r2pk3rpi,0qjtzwvt,eetihg7z,fezyjh7c,gsf6uaa7,j5mpo4f4,k2nta8wd,8ss2ab33,zignlkuw,5xr4362m,alouuudp,5h1ksh0a,ccs5m1xs,ek6arcju,m4gjv5od,ogrdwq8v,zrakrt0e,sofi8jwh,yh7r0cfl,z62ua7bg,0ygumrn2,45a60nz7,h3s3utn7,py0vldv2,q9k8hy5z,x4jajxuv,4rkowt69,baei67zb,f0bymnfe,gok836de,lp0c173d,nf5iccd5,qn82au1c,rgzb6joa,x1sdxtx9,xyfq4o7b,kx0jk99d,v8bqtpr4,z6q2z5ny,zv1n2g5m,xxm3rs77,01n5xm4j,q85sutk9,2rjq84i1,u28eux05,10gxeotk,h2wv4h97,lacfjz5n,i76d65wz,ty8gvk44,z4l94ely"


for run in train_runs:
    train_run_ids.append(run.id)

for run in test_runs:
    orig_train_run_id = run.config["checkpoint"]["wandb_run_id"]
    if orig_train_run_id in train_run_ids:
        train_run_ids.remove(orig_train_run_id)

# for run_id in prev_ids.split(","):
#     if run_id in train_run_ids:
#         train_run_ids.remove(run_id)

print("Untested train run ids:")
print(",".join(train_run_ids))
# for run_id in train_run_ids:
#     print(run_id)
