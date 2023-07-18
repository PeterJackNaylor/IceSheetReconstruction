
slip:
	nextflow run main.nf -params-file config.yaml -resume -profile local

swim:
	nextflow run main_oceans.nf -params-file config_oceans.yaml -resume -profile local