# utils.py

from pytorch_lightning.tuner import Tuner

def model_tuner(
        model, pl_trainer, orig_lr,
        min_lr=1e-10, max_lr=1, test_lr=100,
        lr_finder_mode='exponential', bs_scaler_mode="power",
        steps_per_trial=1, init_bs=2, max_trials=10,
        ):
    tuner = Tuner(pl_trainer)
    # best_batch_size = tuner.scale_batch_size(
    #     model,
    #     mode=bs_scaler_mode,
    #     steps_per_trial=steps_per_trial,
    #     init_val=init_bs,
    #     max_trials=max_trials
    # )
    # model.hparams.batch_size = best_batch_size
    try:
        lr_finder = tuner.lr_find(
            model, min_lr=min_lr, max_lr=max_lr,
            num_training=test_lr, mode=lr_finder_mode
        )
        model.hparams.learning_rate = lr_finder.suggestion()
    except Exception as e:
        print(f"Error during learning rate finder: {e}")
        model.hparams.learning_rate = orig_lr
        return model
    return model
