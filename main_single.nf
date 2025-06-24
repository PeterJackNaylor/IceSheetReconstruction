
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
params_preprocess = file("yaml_configs/" + params.pre_processing_config)

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
params_demcontours = file("yaml_configs/" + params.dem_contours_config)

process DemFile {

    publishDir "${publishdir}/data/DEM", overwrite: true

    input:
        path pydemcontours
        path params
        path polygons_folder
    output:
        path "DEM_Contours.npy"
        path "*.png"
    script:
        """
        python $pydemcontours --params $params --polygons_folder $polygons_folder
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
        each velocity
        path test_file
        path polygons_folder
        path yaml_file

    output:
        path(name)
        path("${name}.csv")
        tuple path("*__trial_scores.csv"), path("multiple")
        path("optuna")

    script:
        name = "INR_${params.data_setup}_Model_${model}_Coh_${coherence}_Swa_${swath}_Dem_${dem}_PDEc_${pde_curve}_velocity_${velocity}"
        """
            python $pyinr --data $data \
                --projection ${params.projection} \
                --name $name \
                --yaml_file $yaml_file \
                --model $model \
                --coherence $coherence \
                --swath $swath \
                --dem $dem --dem_data $dem_file \
                --pde_curv $pde_curve \
                --velocity $velocity \
                --test_file $test_file \
                --polygons_folder $polygons_folder
        """
}
if (["Velocity"].contains(params.data_setup)){
    validation_py = []
    if (params.data_setup == "miniVelocity"){
        validation_py = validation_py + [
            tuple("evaluation/validation_cs2.py", "CS2_2020_test", "all_test_set_cs2.npy"),
            tuple("evaluation/validation_cs2.py", "CS2_2020", "all_cleaned.npy"),
        ]
    }
    if (params.data_setup == "Velocity"){
        validation_py = validation_py + [
            tuple("evaluation/validation_cs2.py", "CS2_2017_2020_test", "all_test_set_cs2.npy"),
            tuple("evaluation/validation_cs2.py", "CS2_2017_2020", "all_cleaned.npy"),
            tuple("evaluation/validation_icebridge.py", "OIB", "oib_within_petermann_ISRIN_time.npy"),
        ]
    }
} else {
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
            tuple("evaluation/validation_icebridge.py", "OIB_medium", "medium_oib.npy"),
        ]
    }
}

filter_data_mask = file("evaluation/filter_with_mask.py")
process FilterExternalValidation {

    input:
        path py_filter
        each val_tag
        path validation_folder
        path polygons_folder

    output:
        tuple val(filepy_validation), val(tag), path("${tag}_mask.npy"), val(dataname)

    script:
        filepy_validation = val_tag[0]
        tag = val_tag[1]
        dataname = val_tag[2]
    """
        python $py_filter --folder $validation_folder \
                                --dataname $dataname \
                                --polygons_folder $polygons_folder \
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
        path polygons_folder

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
                                  --projection ${params.projection} \
                                  --scores_csv $scores \
                                  --multiple_folder $multiple \
                                  --polygons_folder $polygons_folder \
                                  --mask $masks \
                                  --save ${name}___${tag}.csv
    """
}

daily_error_plots = file("postprocessing/daily_error_plots.py")

process DailyErrorPlots {
    publishDir "${publishdir}/model_errors/", overwrite: true, pattern: "*.png"

    input:
        path model_errors
    output:
        path("*.png")
    script:
    """
        python $daily_error_plots
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
        path "${params.name}_beautiful.csv"


    script:
        """
        python $pypublish ${params.name}
        """
}



workflow {

    main:
        test_set = file(params.datapath + "/" + params.test_set_cs2)
        DemFile(pydemcontours, params_demcontours, params.polygons_folder)
        ConvertNC2NPY(pyconvert, params.datapath)
        Pre_process(pypreprocess, ConvertNC2NPY.output, params_preprocess)
        INR(pyinr, Pre_process.out, params.model, params.coherence, params.swath,
            params.dem, DemFile.out[0], params.pde_curve, params.velocity, test_set, params.polygons_folder,
            params_inr)
        FilterExternalValidation(filter_data_mask, validation_py, params.datapath_validation, params.polygons_folder)
        ExternalValidation(FilterExternalValidation.out, params.datapath_validation, INR.out[2], params.polygons_folder)
        DailyErrorPlots(ExternalValidation.out[2])
        RegroupTraining(pyregroup, INR.out[1].collect())
        RegroupValidation(pyregroup_val, ExternalValidation.out[0].groupTuple(by: 0))
        PublishCSV(pypublish, RegroupTraining.out[2], RegroupValidation.out[1].collect())
}
