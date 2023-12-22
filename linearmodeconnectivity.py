# Train the model
trained_model, data_module, trainer = setup_training(  # Assuming these are the names used for the returned objects
    experiment_name="MyExperiment",
    model_type=my_model_type,
    model_hparams=my_model_hparams,
    datamodule_type=my_datamodule_type,
    datamodule_hparams=my_datamodule_hparams,
    max_epochs=10,
    wandb_tags=["tag1", "tag2"]
)

# Train the model
trainer.fit(model, data_module)

# Access trained weights
trained_weights = trained_model.state_dict()