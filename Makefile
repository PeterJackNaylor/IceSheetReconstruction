
slip:
	nextflow run main.nf -params-file config.yaml -resume -profile local

base_model:
	nextflow run main.nf -params-file config.yaml -resume -profile local --coherence Off --swath Off --dem Off --output base_model

coherence_model:
	nextflow run main.nf -params-file config.yaml -resume -profile local --coherence On --swath Off --dem Off --output coherence_model

swath_model:
	nextflow run main.nf -params-file config.yaml -resume -profile local --coherence Off --swath On --dem Off --output swath_model

dem_model:
	nextflow run main.nf -params-file config.yaml -resume -profile local --coherence Off --swath Off --dem On --output dem_model

swim:
	nextflow run main_oceans.nf -params-file config_oceans.yaml -resume -profile local
