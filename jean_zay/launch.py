from pathlib import Path
import os


class JeanZayExperiment:
    def __init__(
        self,
        exp_name,
        job_name,
        slurm_array_nb_jobs=None,
        num_nodes=1,
        num_gpus_per_node=1,
        total_cpus_per_node=64,
        qos="t3",
        account="jik",
        gpu_type="v100",
        cmd_path="train.py",
        time=None,
    ):
        self.expname = exp_name
        self.job_name = job_name
        self.nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.total_cpus = total_cpus_per_node
        self.qos = qos
        self.account = account
        # self.account = "ipk"
        self.gpu_type = gpu_type
        self.slurm_array_nb_jobs = slurm_array_nb_jobs
        self.cmd_path = cmd_path
        self.time = time

    def build_cmd(self, hydra_args):
        hydra_modifiers = []

        for hydra_arg, value in hydra_args.items():
            hydra_modifiers.append(f" {hydra_arg}={value}")
        self.cmd = f"python {self.cmd_path} {''.join(hydra_modifiers)}"
        print(self.cmd)

    def launch(self):
        if not hasattr(self, "cmd"):
            raise ValueError("Run build_cmd first")
        if self.qos == "t4":
            self.qos_name = "qos_gpu-t4"
            self.time = "99:59:59" if self.time is None else self.time
        elif self.qos == "t3":
            self.qos_name = "qos_gpu-t3"
            self.time = "19:59:59" if self.time is None else self.time
        elif self.qos == "dev":
            self.qos_name = "qos_gpu-dev"
            self.time = "01:59:59" if self.time is None else self.time

        else:
            raise ValueError("Not a valid QoS")

        if self.gpu_type == "a100":
            self.gpu_slurm_directive = "#SBATCH -C a100"
            self.total_cpus_per_node = 64

        elif self.gpu_type == "v100":
            self.gpu_slurm_directive = "#SBATCH -C v100-32g"
            self.total_cpus_per_node = 40
        else:
            raise ValueError("Not a valid GPU type")

        self.cpus_per_task = self.total_cpus_per_node // self.num_gpus_per_node

        local_slurmfolder = Path("checkpoints") / Path(self.expname) / Path("slurm")
        local_slurmfolder.mkdir(parents=True, exist_ok=True)
        slurm_path = local_slurmfolder / ("job_file" + ".slurm")
        if type(self.slurm_array_nb_jobs) is int:
            sbatch_array = f"#SBATCH --array=0-{self.slurm_array_nb_jobs-1}"
        elif type(self.slurm_array_nb_jobs) is list:
            sbatch_array = f"#SBATCH --array={','.join([str(i) for i in self.slurm_array_nb_jobs])}"
        elif self.slurm_array_nb_jobs is None:
            sbatch_array = ""
        else:
            raise ValueError("Not a valid type for slurm_array_nb_jobs")
        slurm = f"""#!/bin/bash
#SBATCH --job-name={self.job_name}
{sbatch_array}
#SBATCH --nodes={self.nodes}	# number of nodes
#SBATCH --account={self.account}@{self.gpu_type}
#SBATCH --ntasks-per-node={self.num_gpus_per_node}
#SBATCH --gres=gpu:{self.num_gpus_per_node}
#SBATCH --qos={self.qos_name}
{self.gpu_slurm_directive}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --hint=nomultithread
#SBATCH --time={self.time}
#SBATCH --output=/gpfswork/rech/ufh/ult23zz/plonk/{local_slurmfolder}/job_%j.out
#SBATCH --error=/gpfswork/rech/ufh/ult23zz/plonk/{local_slurmfolder}/job_%j.err
#SBATCH --signal=SIGUSR1@20
module purge
{"module load cpuarch/amd" if self.gpu_type == "a100" else ""}
module load pytorch-gpu/py3/2.0.0
source $WORK/.venvs/plonk/bin/activate

export PYTHONPATH=/gpfswork/rech/ufh/ult23zz/.venvs/diffusion/bin/python
export TRANSFORMERS_OFFLINE=1 # to avoid downloading
export HYDRA_FULL_ERROR=1 # to have the full traceback
export WANDB_CACHE_DIR=$SCRATCH/wandb_cache
export TMPDIR=$JOBSCRATCH
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export IS_CLUSTER=True
set -x
srun {self.cmd}
        """
        with open(slurm_path, "w") as slurm_file:
            slurm_file.write(slurm)

        os.system(f"sbatch {slurm_path}")
