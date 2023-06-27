

dataset = Channel.from(params.dataset)
monthly_dataset = Channel.from(params.monthly_dataset)
datafolder = file(params.dataset_folder)
dataset .map{tuple(it, "")} .concat( monthly_dataset .map{tuple(it, "--no-time")}) .set{ds}
method = Channel.from(params.method)
option = Channel.from(params.normalise)

config = file(params.configname)
pyfile = file("experiments/run.py")


process INR {
    publishDir "nf_meta/", overwrite: true

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
        """
        python ${pyfile} \
            --path ${datafolder}/${data}.npy \
            --name ${name} \
            --yaml_file ${config} \
            --${met}
        """
}


pyfile_group = file("experiments/regroup.py")
process group {
    publishDir "nf_meta/", overwrite: true

    input:
        path(npz)
        
    output:
        path("scores.csv")

    script:
        """
        touch scores.csv
        """
}

workflow {

    main:
        INR(ds, method, option)
        group(INR.output[1].collect())
}