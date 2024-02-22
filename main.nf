
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
println("=                    P.Naylor, A. Stokholm, N. Dionelis                     =")
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

pyconvert = file("pre-processing/data_converter.py")

process ConvertNC2NPY {

    publishDir "outputs/${dataconfig}/data/raw", overwrite: true

    input:
        path pyconvert
        path data
        val dataconfig

    output:
        tuple path("data.npy"), val(dataconfig)

    script:
        """
        python $pyconvert --data_path $data --data_setup $dataconfig
        """
}

pypreprocess = file("pre-processing/preprocess.py")
params_preprocess = file("yaml_configs/" + params.pre_processing_config)

process Pre_process {

    publishDir "outputs/${dataconfig}/data/preprocessed", overwrite: true

    input:
        path pypreprocess
        tuple path(data), val(dataconfig)
        path params
    output:
        tuple path("p-data.npy"), val(dataconfig)

    script:
        """
        python $pypreprocess --data $data --params $params
        """
}

pydemcontours = file("pre-processing/dem_contours.py")
params_demcontours = file("yaml_configs/" + params.dem_contours_config)

process DemFile {

    publishDir "outputs/${dataconfig}/data/DEM", overwrite: true

    input:
        path pydemcontours
        path params
        val dataconfig
    output:
        path "DEM_Contours.npy"
        path "envelop_polygon.pickle"
        path "*.png"
    script:
        """
        python $pydemcontours --params $params
        """
}

if (params.optuna){
    pyinr = file("IceSheetPINNs/optuna_runs.py")
} else {
    pyinr = file("IceSheetPINNs/single_run.py")
}

params_inr = file("yaml_configs/" + params.inr_config)

process INR {

    publishDir "outputs/${dataconfig}/INR", overwrite: true, pattern: "*.pth"
    publishDir "outputs/${dataconfig}/INR", overwrite: true, pattern: "*.npz"
    publishDir "outputs/${dataconfig}/INR", overwrite: true, pattern: "${name}"

    input:
        path pyinr
        tuple path(data), val(dataconfig)
        each coherence
        each swath
        each dem
        path dem_file
        each pde_curve
        path polygon
        path yaml_file


    output:
        tuple path("${name}.pth"), path("${name}.npz"), val(name)
        path(name)
        path("${name}.csv")
        val(dataconfig)

    script:
        name = "INR_${dataconfig}_Coh_${coherence}_Swa_${swath}_Dem_${dem}_PDEc_${pde_curve}"
        """
            python $pyinr --data $data \
                --name $name \
                --yaml_file $yaml_file \
                --coherence $coherence \
                --swath $swath \
                --dem $dem --dem_data $dem_file \
                --pde_curv $pde_curve \
                --polygon $polygon
        """
}

validation_py = [
    tuple("evaluation/validation_icebridge.py", "oib"),
    tuple("evaluation/validation_Geosar.py", "geosar"),
]

process ExternalValidation {
    publishDir "outputs/${dataconfig}/INR/${name}", overwrite: true, pattern: "*.csv"

    input:
        each val_tag
        path validation_folder
        tuple path(weight), path(npz), val(name)
        val dataconfig

    output:
        tuple val(tag), path("${name}___${tag}.csv")

    script:
        validation_method = file(val_tag[0])
        tag = val_tag[1]
    """
        python $validation_method --folder $validation_folder \
                                  --weight $weight --npz $npz \
                                  --save "${name}___${tag}.csv"
    """
}

pyregroup = file("postprocessing/regroup_csv.py")

process RegroupTraining {
    publishDir "outputs/${dataconfig}", overwrite: true
    input:
        path pyregroup
        path csvs
        val dataconfig

    output:
        path "${dataconfig}.csv"

    script:
        """
        python $pyregroup $dataconfig
        """
}

pyregroup_val = file("postprocessing/regroup_csv_validation.py")

process RegroupValidation {
    publishDir "outputs/${dataconfig}", overwrite: true
    input:
        path pyregroup
        tuple val(tag), path(csvs)
        val dataconfig

    output:
        path "${dataconfig}_${tag}.csv"

    script:
        """
        python $pyregroup_val ${dataconfig}_${tag}
        """
}



workflow {

    main:
        DemFile(pydemcontours, params_demcontours, params.data_setup)
        ConvertNC2NPY(pyconvert, params.datapath, params.data_setup)
        Pre_process(pypreprocess, ConvertNC2NPY.output, params_preprocess)
        INR(pyinr, Pre_process.out, params.coherence, params.swath,
            params.dem, DemFile.out[0], params.pde_curve, DemFile.out[1],
            params_inr)
        ExternalValidation(validation_py, params.datapath_validation, INR.out[0], INR.out[3].first())
        RegroupTraining(pyregroup, INR.out[2].collect(), INR.out[3].first())
        RegroupValidation(pyregroup_val, ExternalValidation.out.groupTuple(by: 0), INR.out[3].first())
}
