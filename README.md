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

Then you can:

* Run `utils/test-alex-net-on-server.sh` to copy project code to server and run alex-net demo on server.
* Run `utils/run-command-on-server.sh 'echo y'` to copy project code to server and run command you specified. In this case, it is `echo y`.



## To develop

I have wrapped a docker container for develepment. 

```bash
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
* [neon intrinsic doc](https://developer.arm.com/architectures/instruction-sets/intrinsics/).



## High light

1. Docker development and convenient run in server.
   * This makes development and test environment consistent.
2. Qemu support in docker to enable local test (host x86, but arm elf).
   * Since connection to server is not stable, this makes test can be done locally.
3. Fully tested. Every layer, of simd version or std version, is tested by random generated info and golden data from `torch`.
   * This makes our result reasonable.
4. Simple switch in std and simd code.
   * This ensures that our speedup is achieved by simd version tensor implementation, not by other things.
5. For alex-net test:
   * To reduce time spent by non-layer function, we make a `See` data structure, which reduces time costed on `malloc`.
   * To prevent time difference on other operations, we make lots of functions of `tensor` `static inline`, so in simd version or standard version, these function will spend the same time.
6. Simd impl for every layer.
   * You can compare standard version and simd version directly.



## simd speedup

Compare with standard version and simd version (compiled in Release mode) for one batch alex net infer: 

* in kunpeng 920: (11 sec 261 ms) => (4 sec 170 ms). **Speedup=2.7**
* in macOS M2 Pro: (4 sec 436 ms) => (1 sec 712 ms). **Speedup=2.6**



Simd layer effect:

* After simd relu: 12s => 12s
* After simd fc: 12s => 11s
* After simd softmax: 11s => 11s
* After simd conv2d: 11s => 6s
* After simd maxpool2d: 6s => 5s
