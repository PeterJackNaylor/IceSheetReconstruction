

dataset = Channel.from(params.dataset)
monthly_dataset = Channel.from(params.monthly_dataset)
datafolder = file(params.dataset_folder)
dataset .map{tuple(it, "--time")} .concat( monthly_dataset .map{tuple(it, "--no-time")}) .set{ds}
method = Channel.from(params.method)
option = Channel.from(params.normalise)

config = file(params.configname)
pyfile = file("experiments/run.py")
println(params)

process INR {
    publishDir "${params.output}", overwrite: true

    input:
        tuple val(data), val(opt_2)
        each met
        each opt
        
    output:
        path("${name}.pth")
        path("${name}.npz")

    script:
        if (opt == "normalise"){
            opt = "_${opt}"
            opt2 = "--normalise_targets ${opt_2}"
        } else {
            opt2 = "${opt_2}"
        }
        name = "${met}_${data}${opt}"
        if (params.coherence){
            opt_coherence = " --coherence_path ${datafolder}/${data.replace('data', 'coherence')}.npy"
        }else{
            opt_coherence = ""
        }
        if (params.swath){
            opt_swath = " --swath_path ${datafolder}/${data.replace('data', 'swath')}.npy"
        }else{
            opt_swath = ""
        }
        if (params.swath){
            opt_dem = " --dem_path ${datafolder}/${params.dem_path}"
        }else{
            opt_dem = ""
        }
        """
        python ${pyfile} \
            --path ${datafolder}/${data}.npy \
            --name ${name} \
            --yaml_file ${config} \
            --${met} --gpu ${opt2} ${opt_coherence} ${opt_swath} ${opt_dem}
        """
}


pyfile_group = file("experiments/regroup_l1.py")
process group {
    publishDir "nf_meta/", overwrite: true

    input:
        path(npz)
        
    output:
        path("*.csv")

    script:
        """
        python $pyfile_group
        """
}

workflow {

    main:
        INR(ds, method, option)
        group(INR.output[1].collect())
}
