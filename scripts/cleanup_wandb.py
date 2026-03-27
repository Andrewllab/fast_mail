import wandb

REPO = "nicolasschreiber/pakt_robocasa_pnp_counter_to_stove"
SUBREPO = REPO.split("/")[1]

api = wandb.Api()
collections = [
    coll for coll in api.artifact_type(type_name="model", project=REPO).collections()
]

run_ids = [coll.name.replace("model-", "") for coll in collections]

for idx, run_id in enumerate(run_ids):
    try:
        run = api.run(f"{SUBREPO}/{run_id}")
        artifacts = list(run.logged_artifacts())
        artifact_names = [art.name for art in artifacts]
        artifact_ids = [int(name.split(":v")[1]) for name in artifact_names]

        sorted_artifactnames = [
            val for _, val in sorted(zip(artifact_ids, artifact_names))
        ]
        # sorted_artifacts = [val for _, val in sorted(zip(artifact_ids, artifacts))]

        # print(f"Found run: {run_id}, artifacts: {sorted_artifactnames}")

        if len(artifacts) > 10:
            # keep every fifth artifact from the back since we don't want to delete the most recent ones
            counter = 0
            for i in range(len(artifacts) - 1, 0, -1):
                if counter % 5 != 0:
                    print(f"Deleting artifact: {sorted_artifactnames[i]}")
                    api.artifact(f"{SUBREPO}/{sorted_artifactnames[i]}").delete()
                    pass
                else:
                    print(f"Keeping artifact: {sorted_artifactnames[i]}")
                counter += 1

    except wandb.errors.CommError:
        print(f"Run not found: {run_id}, deleting collection {collections[idx].name}")
        collections[idx].delete()

# for idx, run_id in enumerate(run_ids):
#     api.run(f"3d-sim2real/{run_id}")
#     print(f"Found run: {run_id}")

pass
