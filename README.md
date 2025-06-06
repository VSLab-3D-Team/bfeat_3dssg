
### Project Summary 

![architecture](img/architecture.png)

Most 3D-SSG pipelines lean heavily on GNN reasoning while treating the object embeddings themselves as “good enough.”
Through a detailed error analysis you showed:

- Predicate mistakes explode whenever either subject or object is mis-classified.
- Predicate error rises almost monotonically with the entropy of the object classifier.

Key takeaway: sharpen the object feature space first; the entire graph benefits.

Make sure you set the proper path in "config.py" and project name for wandb in "train_mv_bfeat_ri.py"

### Data Preparation

```bash
python3 preprocess_ply.py
```

### Training Script

```bash
python3 train_mv_bfeat_ri.py --exp_name <experiment name for wandb>
```

### Experiments Script

Experiments for simple MLP classifier with Object Feature Encoder

1. Make sure you set the configuration path in "classifier.py" and "eval_feat_discriminative.py" in experiment directory

2. Run 'eval_feat_discriminative.py' with command below.

```bash
python3 -m experiment.eval_feat_discriminative --exp_dir <Your Path>
```

3. Run "eval_discriminative.py" with command below.

```bash
python3 -m experiment.eval_discriminative --exp_dir <Your Path>
```

4. Run classifier after you configure the "Your Path" part and proper experiment path

```bash
python3 -m experiment.classifier
```

You need to set the 'exp_name' and 'v_exp_name' variable following your path with '.pkl' file extracted in step 2 and 3. 


### Conclusion

By front-loading the pipeline with a truly discriminative, multimodal object encoder—and carefully propagating its semantics through spatially aware, direction-sensitive GNN layers—you set a new state of the art on nearly every 3DSSG benchmark metric, all while offering a drop-in upgrade path for existing models.
