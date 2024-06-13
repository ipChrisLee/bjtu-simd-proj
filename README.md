# bjtu-simd-proj

## To test on server

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

WARNING: YOU SHOULD RUN SCRIPTS IN `utils` IN THE ROOT OF THE PROJECT!

This should print y, which is from your server. At this step, you may need to input password, and if you want not to enter password every time, you can run:

```bash
utils/copy-id-to-server.sh
```

Note: this require you have `~/.ssh/id_ed25519.pub` generated. You can use `ssh-keygen -t ed25519` to generate one.

Then you can run `utils/copy-project-to-remote-and-run.sh` to copy project code to server and run it.

## To develop

I have wrapped a docker container for develepment. 

```bash
# following command should be run at root of project.
docker/build-image.sh
docker/run-container.sh
```

WARNING: YOU SHOULD RUN SCRIPTS IN `docker` IN THE ROOT OF THE PROJECT!

Then you can attach your ide to container (for vscode, use `ms-vscode-remote.remote-containers` is preferred). Code folder is mounted at `/code` inside container. 

For vscode, I have added some suggested plugins in `/code/.vscode/settings.json`, you can install them to have better development experience.

You can also run `docker/goto-container.sh` to bash inside container.

Inside `/code`, you can `utils/build-debug-dev.sh` to build the targets. 

This container has installed qemu, so you can run compiled executable file directly: `build-debug-dev/alex-net`.

## Thanks

* [This doc](https://gist.github.com/luk6xff/9f8d2520530a823944355e59343eadc1) for help on emulating arm elf in another target host.
* [This answer in stackoverflow](https://stackoverflow.com/a/30642130/17924585) for help on how to avoid compile check in cmake toolchain.
* [This wiki article](https://www.armadeus.org/wiki/index.php?title=NEON_HelloWorld) for neon demo.
* [This answer in stackoverflow](https://stackoverflow.com/a/46811527/17924585) for help on how to static link library.
* [This blog](https://ughe.github.io/2018/07/19/qemu-aarch64) for help on how to run elf in qemu-aarch64 when dynamic linking failed.

## High light

1. Docker development and convinient run in server.
2. Qemu support in docker to enable local test (host x86, but arm elf).
3. Fully tested. Every layer is tested.
4. Simple switch in std and simd code.
5. Robust test result. By using `See` data structure and user defined `malloc` to avoid time spend on `malloc` and get precise result. Use `static inline` functions to prevent the effect of unrelated part of program when comparing.
6. Simd impl for every layer.
