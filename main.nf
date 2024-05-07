
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
        path "training_mask.pickle"
        path "validation_mask.pickle"
        path "tight_enveloppe.pickle"
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
    tuple("evaluation/validation_icebridge.py", "OIB", "oib_within_petermann_ISRIN_time.npy"),
    tuple("evaluation/validation_Geosar.py", "GeoSAR", "GeoSAR_Petermann_xband_prep.npy"),
]
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
        // tuple path(filepy_validation), val(tag), path("${tag}_mask.npy")
        tuple val(filepy_validation), val(tag), path("${tag}_mask.npy"), val(dataname)

    script:
        filepy_validation = val_tag[0]
        // filepy_validation = file(val_tag[0])
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
    publishDir "outputs/${dataconfig}/INR/${name}", overwrite: true, pattern: "*.csv"
    publishDir "outputs/${dataconfig}/INR/${name}/${tag}_plots", overwrite: true, pattern: "*.png"

    input:
        each val_tag
        path validation_folder
        tuple path(weight), path(npz), val(name)
        val dataconfig
        path tight_mask
        path train_mask
        path validation_mask

    output:
        tuple val(tag), path("${name}___${tag}.csv")
        path("*.png")

    script:
        validation_method = file(val_tag[0])
        tag = val_tag[1]
        masks = val_tag[2]
        dataname = val_tag[3]
    """
        python $validation_method --folder $validation_folder \
                                  --dataname $dataname \
                                  --weight $weight --npz $npz \
                                  --tight_mask $tight_mask \
                                  --train_mask $train_mask \
                                  --validation_mask $validation_mask \
                                  --mask $masks \
                                  --save ${name}___${tag}.csv
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
        FilterExternalValidation(filter_data_mask, validation_py, params.datapath_validation, DemFile.out[3], DemFile.out[1], DemFile.out[2])
        ExternalValidation(FilterExternalValidation.out, params.datapath_validation, INR.out[0], INR.out[3].first(), DemFile.out[3], DemFile.out[1], DemFile.out[2])
        RegroupTraining(pyregroup, INR.out[2].collect(), INR.out[3].first())
        RegroupValidation(pyregroup_val, ExternalValidation.out[0].groupTuple(by: 0), INR.out[3].first())
}
