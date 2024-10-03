

# Jupyter notebooks on the cluster

1. First connect to the remote on which you want to run the notebook (eg minibaps)

    ssh_minibaps command # if you vave the bash with aliases configured 
    ssh -tAXC leon@84.88.67.74 -p 4022 'ssh -tXC 172.26.20.44' # if no bash


2. Leave the process running without terminal window (ON'T DO THIS STEP IF YOU USE NOHUP LATER)
    screen -S jupyter
    screen -r ??
    screen -list

    
3. select mamba/conda environemnt (sometimes you have to do bash first)
    bash
    mamba activate lab


4. Then on the remote run the jupyter server

    jupyter notebook --no-browser --port=8080
 
*If you want to use NOHUP to leave the process running
    nohup jupyter notebook --no-browser --port=8080 & 


5. Copy the second output link

    Example: http://127.0.0.1:8080/tree?token=668af33892132958827ea7cf65d8230addf151843266a0f2


6. Finally, from your local machine open a terminal and create a tunnel

-   If you can access the remote directly:

    ssh -L 8080:localhost:8080 balma@172.26.20.44

-   Otherwise you need to go through the headnode and forward the ports

    Local machine -> Headnode -> Remote
    8080 -> 8080 -> 8080

    ssh -tL 8080:localhost:8080 balma@84.88.67.74 -p 4022 'ssh -L 8080:localhost:8080 balma@172.26.20.44'


7. Once the tunnel is created copy the link in (3) in your browser


8. To check the nohup output open in home/balma in the cluster open terminal and 
cat nohup.out 

Example output:
Jupyter Notebook 6.4.0 is running at:
[I 12:26:21.145 NotebookApp] http://minibaps:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121
[I 12:26:21.145 NotebookApp]  or http://127.0.0.1:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121
[I 12:26:21.145 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 12:26:21.148 NotebookApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/balma/.local/share/jupyter/runtime/nbserver-346494-open.html
    Or copy and paste one of these URLs:
        http://minibaps:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121
     or http://127.0.0.1:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121

token=668af33892132958827ea7cf65d8230addf151843266a0f2

Important: use the second link (http://127.0.0.1:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121) and  be sure that the port concides with the port of the tunnl in step 6

9. Kill all processes
killall -u balma


### Connect by nautilus to the cluster
sshfs_cluster (to mount directly the closter folder in your pc)


## VSCode on the cluster

1. connect remotely to the cluster using the terminal

2. type codium it will trigger a VSCode window that you can visualize in your current desktop but rund in the cluster


## aliases
cat .bash_aliases
