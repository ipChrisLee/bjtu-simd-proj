# bjtu-simd-proj

## To develop

Create `secret.sh` in project local, this file should contain:

```bash
export remoteServerUser="hbxxxxxx"
export remoteServerAddr="211.xxx"
export proxyJump="proxyjump" # require when you using proxy.
```

Then you can run:

```bash
utils/true-ssh.sh echo y
```

This should print y, which is from your server. At this step, you may need to input password, and if you want not to enter password every time, you can run:

```bash
utils/copy-id-to-server.sh
```

Note: this require you have `~/.ssh/id_ed25519.pub` generated. You can use `ssh-keygen -t ed25519` to generate one.

Then you can run `utils/copy-project-to-remote-and-run.sh` to copy project code to server and run it.

## Thanks

* [This doc](https://gist.github.com/luk6xff/9f8d2520530a823944355e59343eadc1) for help on emulating arm elf in another target host.
* [This answer in stackoverflow](https://stackoverflow.com/a/30642130/17924585) for help on how to avoid compile check in cmake toolchain.
* [This wiki article](https://www.armadeus.org/wiki/index.php?title=NEON_HelloWorld) for neon demo.
* [This answer in stackoverflow](https://stackoverflow.com/a/46811527/17924585) for help on how to static link library.
* [This blog](https://ughe.github.io/2018/07/19/qemu-aarch64) for help on how to run elf in qemu-aarch64 when dynamic linking failed.
