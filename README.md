# krknctl Assist

Install once:

```bash
git clone https://github.com/krkn-chaos/krknctl-assist.git ~/krknctl-assist
cd ~/krknctl-assist
bash ./scripts/install_krknctl_assist.sh
```

Run:

```bash
krknctl-assist
```

Useful options:

```bash
KRKNCTL_BRANCH=pr-149 krknctl-assist
KRKNCTL_DIR=~/tmp/krknctl krknctl-assist --force-build
krknctl-assist --cleanup
krknctl-assist --force-build
```
