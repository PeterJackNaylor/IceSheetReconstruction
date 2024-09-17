
println("=============================================================================")
println("=                                PARAMS:                                    =")
println("=                             ISRIN PIPELINE                                =")
println("=                            INPUT PARAMETERS                               =")
println("=                                                                           =")

params.each{ k, v -> println  params_check(k, v)}

println("=                                                                           =")
println("=                            LISTING COMPLETE                               =")
println("=                         Get ready for take off                            =")
println("=                         Signed:                                           =")
println("=                    P. Naylor, A. Stokholm, N. Dionelis                     =")
println("=                                                                           =")
println("=============================================================================")

def params_check(k, var) {
    underline = "----------------------"
    underline = underline.padRight(75)
    space = " "
    space = space.padRight(75)
    key = "           " + k.padRight(63)
    if (var instanceof String){
        inp = var.padRight(74)
    } else {
        if (var instanceof Boolean){
            if (var){
                inp = "true".padRight(74)
            } else {
                inp = "false".padRight(74)
            }
        } else {
            inp = var.join (', ').padRight(74)
        }
    }
    return "= ${key}=\n=${underline}=\n= ${inp}=\n=${space}="
}

publishdir = "outputs/${params.name}/"
pyconvert = file("pre-processing/data_converter.py")

process ConvertNC2NPY {

    publishDir "${publishdir}/data/raw", overwrite: true

    input:
        path pyconvert
        path data

    output:
        path("data.npy")

    script:
        """
        python $pyconvert --data_path $data --data_setup ${params.data_setup}
        """
}

pypreprocess = file("pre-processing/preprocess.py")
params_preprocess = file("yaml_configs/" + params.name + "/" + params.pre_processing_config)

process Pre_process {

    publishDir "${publishdir}/data/preprocessed", overwrite: true

    input:
        path pypreprocess
        path(data)
        path params
    output:
        path("p-data.npy")

    script:
        """
        python $pypreprocess --data $data --params $params
        """
}

pydemcontours = file("pre-processing/dem_contours.py")
params_demcontours = file("yaml_configs/" + params.name + "/" + params.dem_contours_config)

process DemFile {

    publishDir "${publishdir}/data/DEM", overwrite: true

    input:
        path pydemcontours
        path params
    output:
        path "DEM_Contours.npy"
        path "training_mask.pickle"
        path "validation_mask.pickle"
        path "tight_enveloppe.pickle"
        path "*.png"
    script:
        """
        python $pydemcontours --params $params
        """
}

pyinr = file("IceSheetPINNs/optuna_runs.py")

params_inr = file("yaml_configs/" + params.name + "/" + params.inr_config)

process INR {

    publishDir "${publishdir}/INR/", overwrite: true, pattern: "${name}"
    publishDir "${publishdir}/INR/${name}", overwrite: true, pattern: "*.csv", failOnError: false
    publishDir "${publishdir}/INR/${name}", overwrite: true, pattern: "multiple", failOnError: false
    publishDir "${publishdir}/INR/${name}", overwrite: true, pattern: "optuna", failOnError: false

    input:
        path pyinr
        path(data)
        each model
        each coherence
        each swath
        each dem
        path dem_file
        each pde_curve
        path test_file
        path polygon
        path yaml_file

    output:
        path(name)
        path("${name}.csv")
        tuple path("*__trial_scores.csv"), path("multiple")
        path("optuna")

    script:
        name = "INR_${params.data_setup}_Model_${model}_Coh_${coherence}_Swa_${swath}_Dem_${dem}_PDEc_${pde_curve}"
        """
            python $pyinr --data $data \
                --name $name \
                --yaml_file $yaml_file \
                --model $model \
                --coherence $coherence \
                --swath $swath \
                --dem $dem --dem_data $dem_file \
                --pde_curv $pde_curve \
                --test_file $test_file \
                --polygon $polygon
        """
}

validation_py = [
    tuple("evaluation/validation_icebridge.py", "OIB", "oib_within_petermann_ISRIN_time.npy"),
    tuple("evaluation/validation_Geosar.py", "GeoSAR", "GeoSAR_Petermann_xband_prep.npy"),
]
if (["mini", "small", "medium", "all"].contains(params.data_setup)){
    validation_py = validation_py + [
        tuple("evaluation/validation_cs2.py", "CS2_mini_test", "mini_test_set_cs2.npy"),
        tuple("evaluation/validation_cs2.py", "CS2_mini", "mini_cleaned.npy"),
    ]
}
if (["small", "medium", "all"].contains(params.data_setup)){
    validation_py = validation_py + [
        tuple("evaluation/validation_cs2.py", "CS2_small_test", "small_test_set_cs2.npy"),
        tuple("evaluation/validation_cs2.py", "CS2_small", "small_cleaned.npy"),
    ]
}
if (["medium", "all"].contains(params.data_setup)){
    validation_py = validation_py + [
        tuple("evaluation/validation_cs2.py", "CS2_medium_test", "medium_test_set_cs2.npy"),
        tuple("evaluation/validation_cs2.py", "CS2_medium", "medium_cleaned.npy"),
        tuple("evaluation/validation_icebridge.py", "OIB_small", "mini_small_oib.npy"),
    ]
}
if (params.data_setup == "all"){
    validation_py = validation_py + [
        tuple("evaluation/validation_cs2.py", "CS2_all_test", "all_test_set_cs2.npy"),
        tuple("evaluation/validation_cs2.py", "CS2_all", "all_cleaned.npy"),
        tuple("evaluation/validation_icebridge.py", "OIB_small", "mini_small_oib.npy"),
        tuple("evaluation/validation_icebridge.py", "OIB_medium", "medium_oib.npy"),
    ]
}

filter_data_mask = file("evaluation/filter_with_mask.py")
process FilterExternalValidation {

    input:
        path py_filter
        each val_tag
        path validation_folder
        path tight_mask
        path train_mask
        path validation_mask

    output:
        tuple val(filepy_validation), val(tag), path("${tag}_mask.npy"), val(dataname)

    script:
        filepy_validation = val_tag[0]
        tag = val_tag[1]
        dataname = val_tag[2]
    """
        python $py_filter --folder $validation_folder \
                                --dataname $dataname \
                                --tight_mask $tight_mask \
                                --train_mask $train_mask \
                                --validation_mask $validation_mask \
                                --save ${tag}_mask.npy
    """
}

process ExternalValidation {
    publishDir "${publishdir}/INR/${name}", overwrite: true, pattern: "*.csv"
    publishDir "${publishdir}/INR/${name}/${tag}_plots", overwrite: true, pattern: "*.png"

    input:
        each val_tag
        path validation_folder
        tuple path(scores), path(multiple)
        path tight_mask
        path train_mask
        path validation_mask

    output:
        tuple val(tag), path("${name}___${tag}.csv")
        path("*.png")
        path("${name}___${tag}_model*.csv")

    script:
        validation_method = file(val_tag[0])
        tag = val_tag[1]
        masks = val_tag[2]
        dataname = val_tag[3]
        scores.println()
        name = "$scores".split("__")[0]
    """
        python $validation_method --folder $validation_folder \
                                  --dataname $dataname \
                                  --scores_csv $scores \
                                  --multiple_folder $multiple \
                                  --tight_mask $tight_mask \
                                  --train_mask $train_mask \
                                  --validation_mask $validation_mask \
                                  --mask $masks \
                                  --save ${name}___${tag}.csv
    """
}

pyregroup = file("postprocessing/regroup_csv.py")

process RegroupTraining {
    publishDir publishdir, overwrite: true
    input:
        path pyregroup
        path csvs

    output:
        path "${params.name}.csv"
        path "${params.name}_reduced.csv"
        path "${params.name}_publish.csv"

    script:
        """
        python $pyregroup ${params.name}
        """
}

pyregroup_val = file("postprocessing/regroup_csv_validation.py")
py_clean_val = file("postprocessing/clean_validation_csv.py")

process RegroupValidation {
    publishDir publishdir, overwrite: true
    input:
        path pyregroup
        tuple val(tag), path(csvs)

    output:
        path "${tag}.csv"
        path "${tag}_publish.csv"


    script:
        """
        python $pyregroup_val ${tag}
        python $py_clean_val ${tag}.csv
        """
}

pypublish = file("postprocessing/publish_final_table.py")

process PublishCSV {
    publishDir publishdir, overwrite: true
    input:
        path pyregroup
        path train_csv
        path val__csvs
    output:
        path "published.csv"


    script:
        """
        python $pypublish
        """
}



workflow {

    main:
        test_set = file(params.datapath + "/" + params.test_set_cs2)
        DemFile(pydemcontours, params_demcontours)
        ConvertNC2NPY(pyconvert, params.datapath)
        Pre_process(pypreprocess, ConvertNC2NPY.output, params_preprocess)
        INR(pyinr, Pre_process.out, params.model, params.coherence, params.swath,
            params.dem, DemFile.out[0], params.pde_curve, test_set, DemFile.out[2],
            params_inr)
        FilterExternalValidation(filter_data_mask, validation_py, params.datapath_validation, DemFile.out[3], DemFile.out[1], DemFile.out[2])
        ExternalValidation(FilterExternalValidation.out, params.datapath_validation, INR.out[2], DemFile.out[3], DemFile.out[1], DemFile.out[2])
        RegroupTraining(pyregroup, INR.out[1].collect())
        RegroupValidation(pyregroup_val, ExternalValidation.out[0].groupTuple(by: 0))
        PublishCSV(pypublish, RegroupTraining.out[2], RegroupValidation.out[1].collect())
}
