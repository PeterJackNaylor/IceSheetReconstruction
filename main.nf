
// import groovy.yaml.YamlBuilder

// dataset = Channel.from(params.dataset)
// monthly_dataset = Channel.from(params.monthly_dataset)
// datafolder = file(params.dataset_folder)
// dataset .map{tuple(it, "--time")} .concat( monthly_dataset .map{tuple(it, "--no-time")}) .set{ds}
// method = Channel.from(params.method)
// option = Channel.from(params.normalise)

// config = file(params.configname)

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
        inp = var.join (', ').padRight(74)
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
        path "p-data.npy"

    script:
        """
        python $pypreprocess --data $data --params $params
        """


}

// pyfile = file("experiments/run.py")

// process INR {
//     publishDir "${params.output}", overwrite: true

//     input:
//         tuple val(data), val(opt_2)
//         each met
//         each opt
//         each coherence_opt
//         each swath_opt
//         each dem_opt

//     output:
//         path("${name}.pth")
//         path("${name}.npz")

//     script:
//         if (opt == "normalise"){
//             opt = "_${opt}"
//             opt2 = "--normalise_targets ${opt_2}"
//         } else {
//             opt2 = "${opt_2}"
//         }
//         name = "${met}_${data}${opt}"
//         if (coherence_opt){
//             opt_coherence = " --coherence_path ${datafolder}/${data.replace('data', 'coherence')}.npy"
//         }else{
//             opt_coherence = ""
//         }
//         if (swath_opt){
//             opt_swath = " --swath_path ${datafolder}/${data.replace('data', 'swath')}.npy"
//         }else{
//             opt_swath = ""
//         }
//         if (dem_opt){
//             opt_dem = " --dem_path ${datafolder}/${params.dem_path}"
//         }else{
//             opt_dem = ""
//         }
//         """
//         python ${pyfile} \
//             --path ${datafolder}/${data}.npy \
//             --name ${name} \
//             --yaml_file ${config} \
//             --${met} --gpu ${opt2} ${opt_coherence} ${opt_swath} ${opt_dem}
//         """
// }


// py_evaluate = file("experiments/evaluate.py")

// process Evaluate {
//     publishDir "nf_meta/", overwrite: true

//     input:
//         path(weight)
//         path(npz)
//         path(config)
//     output:
//         path("*.csv")

//     script:
//         """
//         python $py_evaluate --model_param $npz \
//                             --model_weights $weight \
//                             --config ${config} \
//                             --datafolder ${datafolder} \
//                             --support ${datafolder}/${params.support}
//         """
// }

// py_group = file("experiments/group.py")

// process Regroup {
//     publishDir "./", overwrite: true
//     input:
//         path(f)
//     output:
//         path "performance.tsv"
//     script:
//         """
//         python $py_group
//         """

// }

workflow {

    main:
        ConvertNC2NPY(pyconvert, params.datapath, params.data_setup)
        Pre_process(pypreprocess, ConvertNC2NPY.output, params_preprocess)
        // PreProcess(ConvertNC2NPY.output)
        // INR(PreProcess.out, method, option, params.coherence, params.swath, params.dem)
        // Evaluate(INR.output, config)
        // // performance.out.collectFile(name: "performance.tsv", skip: 1, keepHeader: true)
        // Regroup(Evaluate.output.collect())
}
