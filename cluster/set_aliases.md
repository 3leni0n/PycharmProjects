# SSH Aliases and Configuration

This file contains useful SSH aliases for connecting to remote servers efficiently.

## 🔗 What This Setup Allows:
✅ Jumping through an intermediate server to access internal machines  
✅ Setting up an SSH tunnel to forward local ports  
✅ Mounting remote directories via SSHFS  
✅ Using SFTP for file transfers

## Check aliases
```bash
cat .bash_aliases
```

## Edit aliases
```bash
nano .bash_aliases
vim .bash_aliases
```

## SSH Aliases

```bash
# Connect to an intermediate server (84.88.67.74) via SSH, then SSH into 172.26.20.44
alias ssh_minibaps="ssh -tXYC alexis@84.88.67.74 -p 4022 'ssh -tXYC 172.26.20.44'"

# Directly SSH into 172.26.20.44 (assuming you're already connected to the correct network)
alias minibaps="ssh -tXYC 172.26.20.44"

# Similar to ssh_minibaps but connects to a different internal machine (172.26.20.46)
alias ssh_minibaps2="ssh -tXYC alexis@84.88.67.74 -p 4022 'ssh -tXYC 172.26.20.46'"

# Directly SSH into 172.26.20.46
alias minibaps2="ssh -tXYC 172.26.20.46"

# SSH into a Barcelona server (neurocomp.fcrb.es) using port 4022
alias ssh_bcn="ssh -XYC alexis@neurocomp.fcrb.es -p 4022"

# Set up an SSH tunnel: forwards local port 4022 to port 22 (SSH) on the Barcelona server
alias tunneltobcn="ssh -L 4022:127.0.0.1:22 alexis@neurocomp.fcrb.es -p 4022"

# Mount a remote directory from 84.88.67.74 to ~/cluster/ using SSHFS
alias sshfs_cluster="sshfs -p 4022 84.88.67.74:/home/alexis/ ~/cluster/"

# Start an SFTP session with 84.88.67.74 using port 4022
alias sftp_cluster="sftp -p 4022 84.88.67.74"

# Sync files over SSH
alias rsync_ssh='rsync -ahxv --progress -e "ssh -T -p 4022"'
````

## 🔄 Reload Configuration

To apply these aliases, reload your shell configuration:
- For **Bash** users, run:
```bash
source ~/.bashrc
```
- For **Zsh** users, run:
```bash
source ~/.zshrc
```

## ⚙️ Explanation of SSH Options

Here’s what each SSH option does:

| Option  | Description |
|---------|-------------|
| `-t`    | Allocates a pseudo-TTY (needed for running interactive commands) |
| `-X`    | Enables X11 forwarding (allows running GUI applications over SSH) |
| `-Y`    | Enables trusted X11 forwarding (more permissive than `-X`) |
| `-C`    | Enables compression (useful for slow connections) |
| `-p 4022` | Specifies the SSH port (since it’s not using the default port 22) |

🚀 Happy SSH-ing! 🚀
