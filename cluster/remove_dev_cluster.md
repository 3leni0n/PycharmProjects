# Jupyter notebooks on the cluster

1. First connect to the remote on which you want to run the notebook (eg minibaps)
   - **If** you have **Bash** aliases configured:  
       ```bash
       ssh_minibaps
       ```
   - If **not**:
     ```bash
     ssh -tAXC alexis@84.88.67.74 -p 4022 'ssh -tXC 172.26.20.44'
     ```
     
2. Leave the process running without terminal window (***don't do this step if you use 'NOHUP' later**):  
    ```bash
   screen -S jupyter  # Start a new named screen session
   screen -list  # List available sessions
   screen -r jupyter  # Reattach (resume) the session
   ```
   This is useful if your SSH connection drops but you want to resume the process later

3. select conda/mamba environemnt (sometimes you have to do bash first):  
    ```bash
    conda activate env  # For conda
    mamba activate env  # For mamba
   ```

4. Then on the remote run the jupyter server:
    ```bash
    jupyter notebook --no-browser --port=8080
    ```
   - --no-browser → Prevents Jupyter from trying to open a browser (since it's running on a remote server)
   - --port=8080 → Starts Jupyter on port 8080

    **If you want to use NOHUP to leave the process running:**
    ```bash
    nohup jupyter notebook --no-browser --port=8080 &
    ``` 
   - nohup → Runs the command in the background and ignores the hangup signal, preventing it from stopping when you log out  
   - & sends it to the background


5. Copy the second output link:
    ```bash
    Example: http://127.0.0.1:8080/tree?token=668af33892132958827ea7cf65d8230addf151843266a0f2
    ```

6. Finally, from your local machine open a terminal and create a tunnel:
   - If you can access the remote directly:
        ```bash
        ssh -L 8080:localhost:8080 alexis@172.26.20.44
        ```
   - Otherwise you need to go through the headnode and forward the ports:
       Local machine -> Headnode -> Remote  
       8080 -> 8080 -> 8080
       ```bash
       ssh -tL 8080:localhost:8080 alexis@84.88.67.74 -p 4022 'ssh -L 8080:localhost:8080 alexis@172.26.20.44'
       ```

7. Once the tunnel is created copy the link in (3) in your browser

8. To check the **nohup** output open in home/balma in the cluster open terminal and 
cat nohup.out 

Example output:  
Jupyter Notebook 6.4.0 is running at:
[I 12:26:21.145 NotebookApp] http://minibaps:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121  
[I 12:26:21.145 NotebookApp]  or http://127.0.0.1:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121  
[I 12:26:21.145 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation)  
[C 12:26:21.148 NotebookApp]  
    
    To access the notebook, open this file in a browser:
        file:///home/alexis/.local/share/jupyter/runtime/nbserver-346494-open.html
    Or copy and paste one of these URLs:
        http://minibaps:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121
     or http://127.0.0.1:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121

token=668af33892132958827ea7cf65d8230addf151843266a0f2

Important: use the second link (http://127.0.0.1:8081/?token=afc970ebf9a3c718b2cef0fcbcd3164740366595c2bd1121) and  be sure that the port concides with the port of the tunnl in step 6

9. Kill all processes
killall -u alexis

# Connect by nautilus to the cluster
sshfs_cluster (to mount directly the cluster folder in your pc)

# VSCode on the cluster

1. Connect remotely to the cluster using the terminal  
2. Type codium or code to trigger a VSCode window that you can visualize in your current desktop but run in the cluster

# Pycharm on the cluster
Copy this config file -or add its contents to an existing config file- to ~/.ssh/config
```bash
Host cluster
    HostName neurocomp.fcrb.es 
    User alexis
    Port 4022 
    ForwardX11 yes

Host mini
    Hostname 172.26.20.44
    User alexis
    ForwardX11 yes
    ForwardX11Trusted yes
    ProxyCommand ssh -q -W %h:%p cluster

Host mini2
    Hostname 172.26.20.46
    User alexis
    ForwardX11 yes
    ProxyCommand ssh -q -W %h:%p cluster
```

In Pycharm, go to File -> Remote Development -> SSH Connection -> New Connection -> Gear icon  
Alternatively, go to File -> Settings -> SSH Configurations -> Add  
For 'neurocomp.fcrb.es', use the port 4022. For 'minibaps' and 'minibaps2' use the port 22
In 'authentication type' select 'OpenSSH config and authentication agent' and check the box 'Parse OpenSSH config file'
and test the connection