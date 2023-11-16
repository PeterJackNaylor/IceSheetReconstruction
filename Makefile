
slip:
	nextflow run main.nf -params-file yaml_configs/nextflow-config.yaml -resume -profile $(PROFILE)
