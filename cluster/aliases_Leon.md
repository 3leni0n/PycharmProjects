alias ssh_minibaps="ssh -tXYC balma@84.88.67.74 -p 4022 'ssh -tXYC 172.26.20.44$
alias minibaps="ssh -tXYC 172.26.20.44"
alias ssh_minibaps2="ssh -tXYC balma@84.88.67.74 -p 4022 'ssh -tXYC 172.26.20.4$
alias minibaps2="ssh -tXYC 172.26.20.46"
alias ssh_bcn="ssh -XYC balma@neurocomp.fcrb.es -p 4022"
alias tunneltobcn="ssh -L 4022:127.0.0.1:22 balma@neurocomp.fcrb.es -p 4022"
alias sshfs_cluster="sshfs -p 4022 84.88.67.74:/home/balma/ ~/cluster/"
alias sftp_cluster="sftp -p 4022 84.88.67.74"