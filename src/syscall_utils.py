import json
import logging
import re
from collections import defaultdict, Counter
import numpy as np
from typing import Set, List, Dict, Any, Optional
import os
from datetime import datetime
import time
from .config import config

# known_syscalls = set()
# known_arguments = set()
# distinct_syscalls_with_unseen_args = set()


# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

syscall_map = {
    "0": "read", "1": "write", "2": "open", "3": "close", "4": "stat", "5": "fstat", "6": "lstat", "7": "poll",
    "8": "lseek", "9": "mmap",
    "10": "mprotect", "11": "munmap", "12": "brk", "13": "rt_sigaction", "14": "rt_sigprocmask",
    "15": "rt_sigreturn", "16": "ioctl",
    "17": "pread64", "18": "pwrite64", "19": "readv", "20": "writev", "21": "access", "22": "pipe", "23": "select",
    "24": "sched_yield",
    "25": "mremap", "26": "msync", "27": "mincore", "28": "madvise", "29": "shmget", "30": "shmat", "31": "shmctl",
    "32": "dup", "33": "dup2",
    "34": "pause", "35": "nanosleep", "36": "getitimer", "37": "alarm", "38": "setitimer", "39": "getpid",
    "40": "sendfile", "41": "socket",
    "42": "connect", "43": "accept", "44": "sendto", "45": "recvfrom", "46": "sendmsg", "47": "recvmsg",
    "48": "shutdown", "49": "bind",
    "50": "listen", "51": "getsockname", "52": "getpeername", "53": "socketpair", "54": "setsockopt",
    "55": "getsockopt", "56": "clone",
    "57": "fork", "58": "vfork", "59": "execve", "60": "exit", "61": "wait4", "62": "kill", "63": "uname",
    "64": "semget", "65": "semop",
    "66": "semctl", "67": "shmdt", "68": "msgget", "69": "msgsnd", "70": "msgrcv", "71": "msgctl", "72": "fcntl",
    "73": "flock", "74": "fsync",
    "75": "fdatasync", "76": "truncate", "77": "ftruncate", "78": "getdents", "79": "getcwd", "80": "chdir",
    "81": "fchdir", "82": "rename",
    "83": "mkdir", "84": "rmdir", "85": "creat", "86": "link", "87": "unlink", "88": "symlink", "89": "readlink",
    "90": "chmod", "91": "fchmod",
    "92": "chown", "93": "fchown", "94": "lchown", "95": "umask", "96": "gettimeofday", "97": "getrlimit",
    "98": "getrusage", "99": "sysinfo",
    "100": "times", "101": "ptrace", "102": "getuid", "103": "syslog", "104": "getgid", "105": "setuid",
    "106": "setgid", "107": "geteuid",
    "108": "getegid", "109": "setpgid", "110": "getppid", "111": "getpgrp", "112": "setsid", "113": "setreuid",
    "114": "setregid",
    "115": "getgroups", "116": "setgroups", "117": "setresuid", "118": "getresuid", "119": "setresgid",
    "120": "getresgid", "121": "getpgid",
    "122": "setfsuid", "123": "setfsgid", "124": "getsid", "125": "capget", "126": "capset", "127": "rt_sigpending",
    "128": "rt_sigtimedwait",
    "129": "rt_sigqueueinfo", "130": "rt_sigsuspend", "131": "sigaltstack", "132": "utime", "133": "mknod",
    "134": "uselib", "135": "personality",
    "136": "ustat", "137": "statfs", "138": "fstatfs", "139": "sysfs", "140": "getpriority", "141": "setpriority",
    "142": "sched_setparam",
    "143": "sched_getparam", "144": "sched_setscheduler", "145": "sched_getscheduler",
    "146": "sched_get_priority_max", "147": "sched_get_priority_min",
    "148": "sched_rr_get_interval", "149": "mlock", "150": "munlock", "151": "mlockall", "152": "munlockall",
    "153": "vhangup", "154": "modify_ldt",
    "155": "pivot_root", "156": "_sysctl", "157": "prctl", "158": "arch_prctl", "159": "adjtimex",
    "160": "setrlimit", "161": "chroot", "162": "sync",
    "163": "acct", "164": "settimeofday", "165": "mount", "166": "umount2", "167": "swapon", "168": "swapoff",
    "169": "reboot", "170": "sethostname",
    "171": "setdomainname", "172": "iopl", "173": "ioperm", "174": "create_module", "175": "init_module",
    "176": "delete_module", "177": "get_kernel_syms",
    "178": "query_module", "179": "quotactl", "180": "nfsservctl", "181": "getpmsg", "182": "putpmsg",
    "183": "afs_syscall", "184": "tuxcall",
    "185": "security", "186": "gettid", "187": "readahead", "188": "setxattr", "189": "lsetxattr",
    "190": "fsetxattr", "191": "getxattr", "192": "lgetxattr",
    "193": "fgetxattr", "194": "listxattr", "195": "llistxattr", "196": "flistxattr", "197": "removexattr",
    "198": "lremovexattr", "199": "fremovexattr",
    "200": "tkill", "201": "time", "202": "futex", "203": "sched_setaffinity", "204": "sched_getaffinity",
    "205": "set_thread_area", "206": "io_setup",
    "207": "io_destroy", "208": "io_getevents", "209": "io_submit", "210": "io_cancel", "211": "get_thread_area",
    "212": "lookup_dcookie", "213": "epoll_create",
    "214": "epoll_ctl_old", "215": "epoll_wait_old", "216": "remap_file_pages", "217": "getdents64",
    "218": "set_tid_address", "219": "restart_syscall",
    "220": "semtimedop", "221": "fadvise64", "222": "timer_create", "223": "timer_settime", "224": "timer_gettime",
    "225": "timer_getoverrun", "226": "timer_delete",
    "227": "clock_settime", "228": "clock_gettime", "229": "clock_getres", "230": "clock_nanosleep",
    "231": "exit_group", "232": "epoll_wait", "233": "epoll_ctl",
    "234": "tgkill", "235": "utimes", "236": "vserver", "237": "mbind", "238": "set_mempolicy",
    "239": "get_mempolicy", "240": "mq_open", "241": "mq_unlink",
    "242": "mq_timedsend", "243": "mq_timedreceive", "244": "mq_notify", "245": "mq_getsetattr",
    "246": "kexec_load", "247": "waitid", "248": "add_key",
    "249": "request_key", "250": "keyctl", "251": "ioprio_set", "252": "ioprio_get", "253": "inotify_init",
    "254": "inotify_add_watch", "255": "inotify_rm_watch",
    "256": "migrate_pages", "257": "openat", "258": "mkdirat", "259": "mknodat", "260": "fchownat",
    "261": "futimesat", "262": "newfstatat", "263": "unlinkat",
    "264": "renameat", "265": "linkat", "266": "symlinkat", "267": "readlinkat", "268": "fchmodat",
    "269": "faccessat", "270": "pselect6", "271": "ppoll",
    "272": "unshare", "273": "set_robust_list", "274": "get_robust_list", "275": "splice", "276": "tee",
    "277": "sync_file_range", "278": "vmsplice",
    "279": "move_pages", "280": "utimensat", "281": "epoll_pwait", "282": "signalfd", "283": "timerfd_create",
    "284": "eventfd", "285": "fallocate",
    "286": "timerfd_settime", "287": "timerfd_gettime", "288": "accept4", "289": "signalfd4", "290": "eventfd2",
    "291": "epoll_create1", "292": "dup3",
    "293": "pipe2", "294": "inotify_init1", "295": "preadv", "296": "pwritev", "297": "rt_tgsigqueueinfo",
    "298": "perf_event_open", "299": "recvmmsg",
    "300": "fanotify_init", "301": "fanotify_mark", "302": "prlimit64", "303": "name_to_handle_at",
    "304": "open_by_handle_at", "305": "clock_adjtime",
    "306": "syncfs", "307": "sendmmsg", "308": "setns", "309": "getcpu", "310": "process_vm_readv",
    "311": "process_vm_writev", "312": "kcmp", "313": "finit_module",
    "314": "sched_setattr", "315": "sched_getattr", "316": "renameat2", "317": "seccomp", "318": "getrandom",
    "319": "memfd_create", "320": "kexec_file_load",
    "321": "bpf", "322": "execveat", "323": "userfaultfd", "324": "membarrier", "325": "mlock2",
    "326": "copy_file_range", "327": "preadv2", "328": "pwritev2",
    "329": "pkey_mprotect", "330": "pkey_alloc", "331": "pkey_free", "332": "statx", "333": "io_pgetevents",
    "334": "rseq", "424": "pidfd_send_signal",
    "425": "io_uring_setup", "426": "io_uring_enter", "427": "io_uring_register", "428": "open_tree",
    "429": "move_mount", "430": "fsopen", "431": "fsconfig",
    "432": "fsmount", "433": "fspick", "434": "pidfd_open", "435": "clone3", "436": "close_range", "437": "openat2",
    "438": "pidfd_getfd", "439": "faccessat2",
    "440": "process_madvise", "441": "epoll_pwait2", "442": "mount_setattr", "443": "quotactl_fd",
    "444": "landlock_create_ruleset", "445": "landlock_add_rule",
    "446": "landlock_restrict_self", "447": "memfd_secret", "448": "process_mrelease"
}

syscall_arg_format = {
    "read": ["int fd", "void *buf", "size_t count"],
    "write": ["int fd", "const void *buf", "size_t count"],
    "open": ["const char *pathname", "int flags", "mode_t mode"],
    "close": ["int fd"],
    "stat": ["const char *pathname", "struct stat *statbuf"],
    "fstat": ["int fd", "struct stat *statbuf"],
    "lstat": ["const char *pathname", "struct stat *statbuf"],
    "poll": ["struct pollfd *fds", "nfds_t nfds", "int timeout"],
    "lseek": ["int fd", "off_t offset", "int whence"],
    "mmap": ["void *addr", "size_t length", "int prot", "int flags", "int fd", "off_t offset"],
    "mprotect": ["void *addr", "size_t len", "int prot"],
    "munmap": ["void *addr", "size_t length"],
    "brk": ["void *addr"],
    "rt_sigaction": ["int signum", "const struct sigaction *act", "struct sigaction *oldact", "size_t sigsetsize"],
    "rt_sigprocmask": ["int how", "sigset_t *set", "sigset_t *oldset", "size_t sigsetsize"],
    "rt_sigreturn": [],
    "ioctl": ["int fd", "unsigned long request", "unsigned long arg"],
    "pread64": ["int fd", "void *buf", "size_t count", "off_t offset"],
    "pwrite64": ["int fd", "const void *buf", "size_t count", "off_t offset"],
    "readv": ["int fd", "const struct iovec *iov", "int iovcnt"],
    "writev": ["int fd", "const struct iovec *iov", "int iovcnt"],
    "access": ["const char *pathname", "int mode"],
    "pipe": ["int pipefd[2]"],
    "select": ["int nfds", "fd_set *readfds", "fd_set *writefds", "fd_set *exceptfds", "struct timeval *timeout"],
    "sched_yield": [],
    "mremap": ["void *old_address", "size_t old_size", "size_t new_size", "int flags", "void *new_address"],
    "msync": ["void *addr", "size_t length", "int flags"],
    "mincore": ["void *addr", "size_t length", "unsigned char *vec"],
    "madvise": ["void *addr", "size_t length", "int advice"],
    "shmget": ["key_t key", "size_t size", "int shmflg"],
    "shmat": ["int shmid", "const void *shmaddr", "int shmflg"],
    "shmctl": ["int shmid", "int cmd", "struct shmid_ds *buf"],
    "dup": ["int oldfd"],
    "dup2": ["int oldfd", "int newfd"],
    "pause": [],
    "nanosleep": ["const struct timespec *req", "struct timespec *rem"],
    "getitimer": ["int which", "struct itimerval *curr_value"],
    "alarm": ["unsigned int seconds"],
    "setitimer": ["int which", "const struct itimerval *new_value", "struct itimerval *old_value"],
    "getpid": [],
    "sendfile": ["int out_fd", "int in_fd", "off_t *offset", "size_t count"],
    "socket": ["int domain", "int type", "int protocol"],
    "connect": ["int sockfd", "const struct sockaddr *addr", "socklen_t addrlen"],
    "accept": ["int sockfd", "struct sockaddr *addr", "socklen_t *addrlen"],
    "sendto": ["int sockfd", "const void *buf", "size_t len", "int flags", "const struct sockaddr *dest_addr", "socklen_t addrlen"],
    "recvfrom": ["int sockfd", "void *buf", "size_t len", "int flags", "struct sockaddr *src_addr", "socklen_t *addrlen"],
    "sendmsg": ["int sockfd", "const struct msghdr *msg", "int flags"],
    "recvmsg": ["int sockfd", "struct msghdr *msg", "int flags"],
    "shutdown": ["int sockfd", "int how"],
    "bind": ["int sockfd", "const struct sockaddr *addr", "socklen_t addrlen"],
    "listen": ["int sockfd", "int backlog"],
    "getsockname": ["int sockfd", "struct sockaddr *addr", "socklen_t *addrlen"],
    "getpeername": ["int sockfd", "struct sockaddr *addr", "socklen_t *addrlen"],
    "socketpair": ["int domain", "int type", "int protocol", "int sv[2]"],
    "setsockopt": ["int sockfd", "int level", "int optname", "const void *optval", "socklen_t optlen"],
    "getsockopt": ["int sockfd", "int level", "int optname", "void *optval", "socklen_t *optlen"],
    "clone": ["int flags", "void *child_stack", "int *parent_tid", "int *child_tid", "unsigned long tls"],
    "fork": [],
    "vfork": [],
    "execve": ["const char *filename", "char *const argv[]", "char *const envp[]"],
    "exit": ["int status"],
    "wait4": ["pid_t pid", "int *wstatus", "int options", "struct rusage *rusage"],
    "kill": ["pid_t pid", "int sig"],
    "uname": ["struct utsname *buf"],
    "semget": ["key_t key", "int nsems", "int semflg"],
    "semop": ["int semid", "struct sembuf *sops", "size_t nsops"],
    "semctl": ["int semid", "int semnum", "int cmd", "union semun arg"],
    "shmdt": ["const void *shmaddr"],
    "msgget": ["key_t key", "int msgflg"],
    "msgsnd": ["int msqid", "const void *msgp", "size_t msgsz", "int msgflg"],
    "msgrcv": ["int msqid", "void *msgp", "size_t msgsz", "long msgtyp", "int msgflg"],
    "msgctl": ["int msqid", "int cmd", "struct msqid_ds *buf"],
    "fcntl": ["int fd", "int cmd", "unsigned long arg"],
    "flock": ["int fd", "int operation"],
    "fsync": ["int fd"],
    "fdatasync": ["int fd"],
    "truncate": ["const char *path", "off_t length"],
    "ftruncate": ["int fd", "off_t length"],
    "getdents": ["unsigned int fd", "struct linux_dirent *dirp", "unsigned int count"],
    "getcwd": ["char *buf", "size_t size"],
    "chdir": ["const char *path"],
    "fchdir": ["int fd"],
    "rename": ["const char *oldpath", "const char *newpath"],
    "mkdir": ["const char *pathname", "mode_t mode"],
    "rmdir": ["const char *pathname"],
    "creat": ["const char *pathname", "mode_t mode"],
    "link": ["const char *oldpath", "const char *newpath"],
    "unlink": ["const char *pathname"],
    "symlink": ["const char *target", "const char *linkpath"],
    "readlink": ["const char *pathname", "char *buf", "size_t bufsiz"],
    "chmod": ["const char *pathname", "mode_t mode"],
    "fchmod": ["int fd", "mode_t mode"],
    "chown": ["const char *pathname", "uid_t owner", "gid_t group"],
    "fchown": ["int fd", "uid_t owner", "gid_t group"],
    "lchown": ["const char *pathname", "uid_t owner", "gid_t group"],
    "umask": ["mode_t mask"],
    "gettimeofday": ["struct timeval *tv", "struct timezone *tz"],
    "getrlimit": ["int resource", "struct rlimit *rlim"],
    "getrusage": ["int who", "struct rusage *usage"],
    "sysinfo": ["struct sysinfo *info"],
    "times": ["struct tms *tbuf"],
    "ptrace": ["long request", "pid_t pid", "void *addr", "void *data"],
    "getuid": [],
    "syslog": ["int type", "char *bufp", "int len"],
    "getgid": [],
    "setuid": ["uid_t uid"],
    "setgid": ["gid_t gid"],
    "geteuid": [],
    "getegid": [],
    "setpgid": ["pid_t pid", "pid_t pgid"],
    "getppid": [],
    "getpgrp": [],
    "setsid": [],
    "setreuid": ["uid_t ruid", "uid_t euid"],
    "setregid": ["gid_t rgid", "gid_t egid"],
    "getgroups": ["int size", "gid_t list[]"],
    "setgroups": ["int size", "const gid_t *list"],
    "setresuid": ["uid_t ruid", "uid_t euid", "uid_t suid"],
    "getresuid": ["uid_t *ruid", "uid_t *euid", "uid_t *suid"],
    "setresgid": ["gid_t rgid", "gid_t egid", "gid_t sgid"],
    "getresgid": ["gid_t *rgid", "gid_t *egid", "gid_t *sgid"],
    "getpgid": ["pid_t pid"],
    "setfsuid": ["uid_t fsuid"],
    "setfsgid": ["gid_t fsgid"],
    "getsid": ["pid_t pid"],
    "capget": ["cap_user_header_t header", "cap_user_data_t dataptr"],
    "capset": ["cap_user_header_t header", "const cap_user_data_t dataptr"],
    "rt_sigpending": ["sigset_t *set", "size_t sigsetsize"],
    "rt_sigtimedwait": ["const sigset_t *uthese", "siginfo_t *uinfo", "const struct timespec *uts", "size_t sigsetsize"],
    "rt_sigqueueinfo": ["pid_t tgid", "int sig", "siginfo_t *uinfo"],
    "rt_sigsuspend": ["sigset_t *unewset", "size_t sigsetsize"],
    "sigaltstack": ["const struct sigaltstack *uss", "struct sigaltstack *uoss"],
    "utime": ["const char *filename", "const struct utimbuf *times"],
    "mknod": ["const char *pathname", "mode_t mode", "dev_t dev"],
    "uselib": ["const char *library"],
    "personality": ["unsigned long persona"],
    "ustat": ["dev_t dev", "struct ustat *ubuf"],
    "statfs": ["const char *path", "struct statfs *buf"],
    "fstatfs": ["int fd", "struct statfs *buf"],
    "sysfs": ["int option"],
    "getpriority": ["int which", "int who"],
    "setpriority": ["int which", "int who", "int prio"],
    "sched_setparam": ["pid_t pid", "const struct sched_param *param"],
    "sched_getparam": ["pid_t pid", "struct sched_param *param"],
    "sched_setscheduler": ["pid_t pid", "int policy", "const struct sched_param *param"],
    "sched_getscheduler": ["pid_t pid"],
    "sched_get_priority_max": ["int policy"],
    "sched_get_priority_min": ["int policy"],
    "sched_rr_get_interval": ["pid_t pid", "struct timespec *interval"],
    "mlock": ["const void *addr", "size_t len"],
    "munlock": ["const void *addr", "size_t len"],
    "mlockall": ["int flags"],
    "munlockall": [],
    "vhangup": [],
    "modify_ldt": ["int func", "void *ptr", "unsigned long bytecount"],
    "pivot_root": ["const char *new_root", "const char *put_old"],
    "_sysctl": ["struct __sysctl_args *args"],
    "prctl": ["int option", "unsigned long arg2", "unsigned long arg3", "unsigned long arg4", "unsigned long arg5"],
    "arch_prctl": ["int code", "unsigned long addr"],
    "adjtimex": ["struct timex *buf"],
    "setrlimit": ["int resource", "const struct rlimit *rlim"],
    "chroot": ["const char *path"],
    "sync": [],
    "acct": ["const char *filename"],
    "settimeofday": ["const struct timeval *tv", "const struct timezone *tz"],
    "mount": ["const char *source", "const char *target", "const char *filesystemtype", "unsigned long mountflags", "const void *data"],
    "umount2": ["const char *target", "int flags"],
    "swapon": ["const char *path", "int swapflags"],
    "swapoff": ["const char *path"],
    "reboot": ["int magic1", "int magic2", "unsigned int cmd", "void *arg"],
    "sethostname": ["const char *name", "size_t len"],
    "setdomainname": ["const char *name", "size_t len"],
    "iopl": ["int level"],
    "ioperm": ["unsigned long from", "unsigned long num", "int turn_on"],
    "init_module": ["void *umod", "unsigned long len", "const char *uargs"],
    "delete_module": ["const char *name_user", "unsigned int flags"],
    "quotactl": ["unsigned int cmd", "const char *special", "int id", "void *addr"],
    "gettid": [],
    "readahead": ["int fd", "off64_t offset", "size_t count"],
    "setxattr": ["const char *path", "const char *name", "const void *value", "size_t size", "int flags"],
    "lsetxattr": ["const char *path", "const char *name", "const void *value", "size_t size", "int flags"],
    "fsetxattr": ["int fd", "const char *name", "const void *value", "size_t size", "int flags"],
    "getxattr": ["const char *path", "const char *name", "void *value", "size_t size"],
    "lgetxattr": ["const char *path", "const char *name", "void *value", "size_t size"],
    "fgetxattr": ["int fd", "const char *name", "void *value", "size_t size"],
    "listxattr": ["const char *path", "char *list", "size_t size"],
    "llistxattr": ["const char *path", "char *list", "size_t size"],
    "flistxattr": ["int fd", "char *list", "size_t size"],
    "removexattr": ["const char *path", "const char *name"],
    "lremovexattr": ["const char *path", "const char *name"],
    "fremovexattr": ["int fd", "const char *name"],
    "tkill": ["pid_t pid", "int sig"],
    "time": ["time_t *tloc"],
    "futex": ["u32 *uaddr", "int op", "u32 val", "struct __kernel_timespec *utime", "u32 *uaddr2", "u32 val3"],
    "sched_setaffinity": ["pid_t pid", "size_t cpusetsize", "unsigned long *mask"],
    "sched_getaffinity": ["pid_t pid", "size_t cpusetsize", "unsigned long *mask"],
    "set_thread_area": ["struct user_desc *u_info"],
    "io_setup": ["unsigned nr_events", "aio_context_t *ctx_idp"],
    "io_destroy": ["aio_context_t ctx"],
    "io_getevents": ["aio_context_t ctx_id", "long min_nr", "long nr", "struct io_event *events", "struct timespec *timeout"],
    "io_submit": ["aio_context_t ctx_id", "long nr", "struct iocb **iocbpp"],
    "io_cancel": ["aio_context_t ctx_id", "struct iocb *iocb", "struct io_event *result"],
    "get_thread_area": ["struct user_desc *u_info"],
    "lookup_dcookie": ["u64 cookie64", "char *buf", "size_t len"],
    "epoll_create": ["int size"],
    "epoll_ctl_old": ["int epfd", "int op", "int fd", "struct epoll_event *event"],
    "epoll_wait_old": ["int epfd", "struct epoll_event *events", "int maxevents", "int timeout"],
    "remap_file_pages": ["void *start", "size_t size", "int prot", "size_t pgoff", "int flags"],
    "getdents64": ["unsigned int fd", "struct linux_dirent64 *dirp", "unsigned int count"],
    "set_tid_address": ["int *tidptr"],
    "restart_syscall": [],
    "semtimedop": ["int semid", "struct sembuf *sops", "unsigned nsops", "const struct timespec *timeout"],
    "fadvise64": ["int fd", "off_t offset", "off_t len", "int advice"],
    "timer_create": ["const clockid_t clockid", "struct sigevent *sevp", "timer_t *timerid"],
    "timer_settime": ["timer_t timerid", "int flags", "const struct itimerspec *new_value", "struct itimerspec *old_value"],
    "timer_gettime": ["timer_t timerid", "struct itimerspec *curr_value"],
    "timer_getoverrun": ["timer_t timerid"],
    "timer_delete": ["timer_t timerid"],
    "clock_settime": ["const clockid_t clockid", "const struct timespec *tp"],
    "clock_gettime": ["const clockid_t clockid", "struct timespec *tp"],
    "clock_getres": ["const clockid_t clockid", "struct timespec *res"],
    "clock_nanosleep": ["const clockid_t clockid", "int flags", "const struct timespec *rqtp", "struct timespec *rmtp"],
    "exit_group": ["int status"],
    "epoll_wait": ["int epfd", "struct epoll_event *events", "int maxevents", "int timeout"],
    "epoll_ctl": ["int epfd", "int op", "int fd", "struct epoll_event *event"],
    "tgkill": ["pid_t tgid", "pid_t pid", "int sig"],
    "utimes": ["const char *filename", "const struct timeval times[2]"],
    "vserver": [],
    "mbind": ["void *start", "unsigned long len", "int mode", "const unsigned long *nodemask", "unsigned long maxnode", "unsigned flags"],
    "set_mempolicy": ["int mode", "const unsigned long *nodemask", "unsigned long maxnode"],
    "get_mempolicy": ["int *policy", "unsigned long *nodemask", "unsigned long maxnode", "unsigned long addr", "unsigned long flags"],
    "mq_open": ["const char *name", "int oflag", "mode_t mode", "struct mq_attr *attr"],
    "mq_unlink": ["const char *name"],
    "mq_timedsend": ["mqd_t mqdes", "const char *msg_ptr", "size_t msg_len", "unsigned msg_prio", "const struct timespec *abs_timeout"],
    "mq_timedreceive": ["mqd_t mqdes", "char *msg_ptr", "size_t msg_len", "unsigned *msg_prio", "const struct timespec *abs_timeout"],
    "mq_notify": ["mqd_t mqdes", "const struct sigevent *sevp"],
    "mq_getsetattr": ["mqd_t mqdes", "const struct mq_attr *newattr", "struct mq_attr *oldattr"],
    "kexec_load": ["unsigned long entry", "unsigned long nr_segments", "struct kexec_segment *segments", "unsigned long flags"],
    "waitid": ["idtype_t idtype", "id_t id", "siginfo_t *infop", "int options", "struct rusage *rusage"],
    "add_key": ["const char *_type", "const char *_description", "const void *_payload", "size_t plen", "key_serial_t keyring"],
    "request_key": ["const char *_type", "const char *_description", "const char *_callout_info", "key_serial_t dest_keyring"],
    "keyctl": ["int option", "unsigned long arg2", "unsigned long arg3", "unsigned long arg4", "unsigned long arg5"],
    "ioprio_set": ["int which", "int who", "int ioprio"],
    "ioprio_get": ["int which", "int who"],
    "inotify_init": [],
    "inotify_add_watch": ["int fd", "const char *pathname", "uint32_t mask"],
    "inotify_rm_watch": ["int fd", "int wd"],
    "migrate_pages": ["pid_t pid", "unsigned long maxnode", "const unsigned long *old_nodes", "const unsigned long *new_nodes"],
    "openat": ["int dfd", "const char *filename", "int flags", "mode_t mode"],
    "mkdirat": ["int dfd", "const char *pathname", "mode_t mode"],
    "mknodat": ["int dfd", "const char *filename", "mode_t mode", "dev_t dev"],
    "fchownat": ["int dfd", "const char *filename", "uid_t user", "gid_t group", "int flag"],
    "futimesat": ["int dfd", "const char *filename", "const struct timeval *utimes"],
    "newfstatat": ["int dfd", "const char *filename", "struct stat *statbuf", "int flag"],
    "unlinkat": ["int dfd", "const char *pathname", "int flag"],
    "renameat": ["int olddfd", "const char *oldname", "int newdfd", "const char *newname"],
    "linkat": ["int olddfd", "const char *oldname", "int newdfd", "const char *newname", "int flags"],
    "symlinkat": ["const char *oldname", "int newdfd", "const char *newname"],
    "readlinkat": ["int dfd", "const char *pathname", "char *buf", "int bufsiz"],
    "fchmodat": ["int dfd", "const char *filename", "mode_t mode"],
    "faccessat": ["int dfd", "const char *filename", "int mode"],
    "pselect6": ["int n", "fd_set *inp", "fd_set *outp", "fd_set *exp", "const struct timespec *tsp", "const sigset_t *sig"],
    "ppoll": ["struct pollfd *ufds", "unsigned int nfds", "const struct timespec *tsp", "const sigset_t *sigmask", "size_t sigsetsize"],
    "unshare": ["int flags"],
    "set_robust_list": ["struct robust_list_head *head", "size_t len"],
    "get_robust_list": ["int pid", "struct robust_list_head **head_ptr", "size_t *len_ptr"],
    "splice": ["int fd_in", "off_t *off_in", "int fd_out", "off_t *off_out", "size_t len", "unsigned int flags"],
    "tee": ["int fdin", "int fdout", "size_t len", "unsigned int flags"],
    "sync_file_range": ["int fd", "off64_t offset", "off64_t nbytes", "unsigned int flags"],
    "vmsplice": ["int fd", "const struct iovec *iov", "unsigned long nr_segs", "unsigned int flags"],
    "move_pages": ["pid_t pid", "unsigned long nr_pages", "const void **pages", "const int *nodes", "int *status", "int flags"],
    "utimensat": ["int dfd", "const char *pathname", "const struct timespec *utimes", "int flags"],
    "epoll_pwait": ["int epfd", "struct epoll_event *events", "int maxevents", "int timeout", "const sigset_t *sigmask", "size_t sigsetsize"],
    "signalfd": ["int ufd", "sigset_t *user_mask", "size_t sizemask"],
    "timerfd_create": ["int clockid", "int flags"],
    "eventfd": ["unsigned int count"],
    "fallocate": ["int fd", "int mode", "off_t offset", "off_t len"],
    "timerfd_settime": ["int ufd", "int flags", "const struct itimerspec *utmr", "struct itimerspec *otmr"],
    "timerfd_gettime": ["int ufd", "struct itimerspec *otmr"],
    "accept4": ["int fd", "struct sockaddr *upeer_sockaddr", "int *upeer_addrlen", "int flags"],
    "signalfd4": ["int ufd", "sigset_t *user_mask", "size_t sizemask", "int flags"],
    "eventfd2": ["unsigned int count", "int flags"],
    "epoll_create1": ["int flags"],
    "dup3": ["int oldfd", "int newfd", "int flags"],
    "pipe2": ["int pipefd[2]", "int flags"],
    "inotify_init1": ["int flags"],
    "preadv": ["int fd", "const struct iovec *vec", "int vlen", "off_t pos_l", "off_t pos_h"],
    "pwritev": ["int fd", "const struct iovec *vec", "int vlen", "off_t pos_l", "off_t pos_h"],
    "rt_tgsigqueueinfo": ["pid_t tgid", "pid_t pid", "int sig", "siginfo_t *uinfo"],
    "perf_event_open": ["struct perf_event_attr *attr_uptr", "pid_t pid", "int cpu", "int group_fd", "unsigned long flags"],
    "recvmmsg": ["int fd", "struct mmsghdr *mmsg", "unsigned int vlen", "unsigned int flags", "struct timespec *timeout"],
    "fanotify_init": ["unsigned int flags", "unsigned int event_f_flags"],
    "fanotify_mark": ["int fanotify_fd", "unsigned int flags", "uint64_t mask", "int dfd", "const char *pathname"],
    "prlimit64": ["pid_t pid", "unsigned int resource", "const struct rlimit64 *new_rlim", "struct rlimit64 *old_rlim"],
    "name_to_handle_at": ["int dfd", "const char *name", "struct file_handle *handle", "int *mnt_id", "int flag"],
    "open_by_handle_at": ["int mountdirfd", "struct file_handle *handle", "int flags"],
    "clock_adjtime": ["const clockid_t which_clock", "struct timex *utx"],
    "syncfs": ["int fd"],
    "sendmmsg": ["int fd", "struct mmsghdr *mmsg", "unsigned int vlen", "unsigned int flags"],
    "setns": ["int fd", "int nstype"],
    "getcpu": ["unsigned *cpu", "unsigned *node", "struct getcpu_cache *tcache"],
    "process_vm_readv": ["pid_t pid", "const struct iovec *lvec", "unsigned long liovcnt", "const struct iovec *rvec", "unsigned long riovcnt", "unsigned long flags"],
    "process_vm_writev": ["pid_t pid", "const struct iovec *lvec", "unsigned long liovcnt", "const struct iovec *rvec", "unsigned long riovcnt", "unsigned long flags"],
    "kcmp": ["pid_t pid1", "pid_t pid2", "int type", "unsigned long idx1", "unsigned long idx2"],
    "finit_module": ["int fd", "const char *uargs", "int flags"],
    "sched_setattr": ["pid_t pid", "struct sched_attr *attr", "unsigned int flags"],
    "sched_getattr": ["pid_t pid", "struct sched_attr *attr", "unsigned int size", "unsigned int flags"],
    "renameat2": ["int olddfd", "const char *oldname", "int newdfd", "const char *newname", "unsigned int flags"],
    "seccomp": ["unsigned int op", "unsigned int flags", "const void *uargs"],
    "getrandom": ["void *buf", "size_t count", "unsigned int flags"],
    "memfd_create": ["const char *uname", "unsigned int flags"],
    "bpf": ["int cmd", "union bpf_attr *attr", "unsigned int size"],
    "execveat": ["int dfd", "const char *filename", "char *const argv[]", "char *const envp[]", "int flags"],
    "userfaultfd": ["int flags"],
    "membarrier": ["int cmd", "int flags"],
    "mlock2": ["const void *start", "size_t len", "int flags"],
    "copy_file_range": ["int fd_in", "off64_t *off_in", "int fd_out", "off64_t *off_out", "size_t len", "unsigned int flags"],
    "preadv2": ["int fd", "const struct iovec *vec", "int vlen", "off_t pos_l", "off_t pos_h", "int flags"],
    "pwritev2": ["int fd", "const struct iovec *vec", "int vlen", "off_t pos_l", "off_t pos_h", "int flags"],
    "pkey_mprotect": ["void *start", "size_t len", "int prot", "int pkey"],
    "pkey_alloc": ["unsigned int flags", "unsigned int init_val"],
    "pkey_free": ["int pkey"],
    "statx": ["int dfd", "const char *filename", "unsigned int flags", "unsigned int mask", "struct statx *buffer"]
}


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global sets for tracking syscalls and arguments
known_syscalls: Set[str] = set()
known_arguments: Set[str] = set()
distinct_syscalls_with_unseen_args: Set[str] = set()


# # ANSI color codes for console output
# RED = '\033[91m'
# GREEN = '\033[92m'
# BLUE = '\033[94m'
# YELLOW = '\033[93m'
# RESET = '\033[0m'

# Enhanced syscall mapping with categories
# syscall_categories = {
#     'file_operations': ['read', 'write', 'open', 'close', 'stat', 'fstat', 'lstat', 'access', 'chmod', 'chown'],
#     'process_management': ['fork', 'clone', 'execve', 'exit', 'wait4', 'kill'],
#     'memory_management': ['mmap', 'munmap', 'mprotect', 'brk'],
#     'network_operations': ['socket', 'connect', 'bind', 'listen', 'accept', 'send', 'recv'],
#     'system_info': ['uname', 'getpid', 'gettid', 'getuid', 'getgid'],
#     'ipc': ['pipe', 'socket', 'msgget', 'msgsnd', 'msgrcv']
# }

syscall_categories = {
    'file_operations': [
        'read', 'write', 'open', 'close', 'stat', 'fstat', 'lstat',
        'access', 'chmod', 'chown', 'rename', 'unlink', 'mkdir', 'rmdir',
        'truncate', 'ftruncate', 'symlink', 'link', 'readlink', 'utime'
    ],
    'process_management': [
        'fork', 'clone', 'execve', 'exit', 'wait4', 'kill', 'setpgid',
        'getpgid', 'getpriority', 'setpriority', 'setsid', 'ptrace', 'prlimit64'
    ],
    'memory_management': [
        'mmap', 'munmap', 'mprotect', 'brk', 'mremap', 'shmat',
        'shmdt', 'shmget', 'shmctl', 'mlock', 'munlock', 'madvise'
    ],
    'network_operations': [
        'socket', 'connect', 'bind', 'listen', 'accept', 'send',
        'recv', 'sendto', 'recvfrom', 'shutdown', 'setsockopt', 'getsockopt',
        'getpeername', 'getsockname', 'sendmsg', 'recvmsg'
    ],
    'system_info': [
        'uname', 'getpid', 'gettid', 'getuid', 'getgid', 'getppid',
        'sysinfo', 'getresuid', 'getresgid', 'gethostname', 'getdomainname'
    ],
    'ipc': [
        'pipe', 'socket', 'msgget', 'msgsnd', 'msgrcv', 'semget',
        'semop', 'semtimedop', 'shmat', 'shmdt', 'shmget', 'shmctl'
    ],
    'device_management': [
        'ioctl', 'ioperm', 'iopl', 'inb', 'outb', 'reboot',
        'sysfs', 'syslog', 'perf_event_open'
    ],
    'time_management': [
        'time', 'stime', 'gettimeofday', 'settimeofday', 'clock_gettime',
        'clock_settime', 'clock_getres', 'nanosleep', 'alarm', 'timer_create',
        'timer_settime', 'timer_gettime', 'timer_delete'
    ],
    'user_management': [
        'setuid', 'setgid', 'setreuid', 'setregid', 'seteuid', 'setegid',
        'setgroups', 'setresuid', 'setresgid', 'capset', 'capget'
    ],
    'audit_and_security': [
        'seccomp', 'keyctl', 'add_key', 'request_key', 'capset',
        'capget', 'ptrace', 'audit', 'quotactl', 'setns'
    ],
    'storage_management': [
        'mount', 'umount', 'pivot_root', 'swapon', 'swapoff',
        'sync', 'fsync', 'fdatasync', 'syncfs'
    ],
    'virtualization': [
        'clone', 'setns', 'unshare', 'ioctl', 'kvm_run', 'kvm_set_msrs',
        'kvm_get_msrs', 'kvm_set_cpuid2', 'kvm_create_vm', 'kvm_destroy_vm'
    ],
    'kernel_management': [
        'sysctl', 'kexec_load', 'kexec_file_load', 'pivot_root',
        'sysfs', 'syslog', 'init_module', 'delete_module'
    ],
    'resource_management': [
        'getrlimit', 'setrlimit', 'prlimit64', 'getrusage',
        'sched_getaffinity', 'sched_setaffinity', 'sched_yield'
    ],
    'file_system_operations': [
        'statfs', 'fstatfs', 'stat', 'fstat', 'mkdir', 'rmdir', 'truncate',
        'ftruncate', 'rename', 'symlink', 'readlink', 'unlink', 'chmod'
    ],
    'module_management': [
        'init_module', 'delete_module', 'finit_module', 'query_module'
    ],
    'signal_management': [
        'signal', 'sigaction', 'sigprocmask', 'sigsuspend', 'sigwaitinfo',
        'sigqueue', 'rt_sigaction', 'rt_sigprocmask', 'rt_sigreturn'
    ],
    'system_shutdown': [
        'reboot', 'shutdown', 'halt', 'poweroff'
    ],
    'trace_and_debug': [
        'ptrace', 'prctl', 'seccomp', 'process_vm_readv',
        'process_vm_writev', 'bpf', 'perf_event_open'
    ],
    'user_session_management': [
        'setsid', 'setpgid', 'getpgid', 'setpgrp', 'getpgrp'
    ],
    'mutex_and_lock_operations': [
        'futex', 'semop', 'semget', 'semtimedop', 'set_robust_list', 'get_robust_list'
    ],
    'queue_management': [
        'mq_open', 'mq_unlink', 'mq_send', 'mq_receive',
        'mq_notify', 'mq_getsetattr'
    ],
    'randomness_operations': [
        'getrandom', 'random', 'srandom', 'initstate', 'setstate'
    ],
    'semaphore_management': [
        'semget', 'semop', 'semctl', 'semtimedop'
    ],
    'resource_limits': [
        'getrlimit', 'setrlimit', 'getpriority', 'setpriority'
    ],
    'character_device_operations': [
        'read', 'write', 'ioctl', 'open', 'close'
    ],
    'special_file_operations': [
        'mknod', 'mknodat', 'fchdir', 'chroot'
    ],
    'database_and_logging': [
        'openlog', 'syslog', 'closelog'
    ]
}



# # Enhanced argument patterns for validation
# argument_patterns = {
#     'file_path': r'^(/[^/\0]+)+$',
#     'number': r'^-?\d+$',
#     'hex': r'^0x[0-9a-fA-F]+$',
#     'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
#     'port': r'^\d{1,5}$'
# }

# Enhanced argument patterns for validation
argument_patterns = {
    'file_path': r'^(/[^/\0]+)+$',                # Validates absolute file paths
    'file_name': r'^[\w,\s-]+\.[A-Za-z]{2,4}$',   # Validates filenames with extensions
    'directory_path': r'^(/[^/\0]*)*$',           # Validates directory paths
    'number': r'^-?\d+$',                         # Validates integers (positive/negative)
    'unsigned_number': r'^\d+$',                  # Validates unsigned integers
    'float': r'^-?\d+(\.\d+)?$',                  # Validates floating-point numbers
    'hex': r'^0x[0-9a-fA-F]+$',                   # Validates hexadecimal format
    'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',  # Validates IPv4 addresses
    'ipv6_address': r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$', # Validates IPv6 addresses
    'port': r'^\d{1,5}$',                         # Validates port numbers (0–65535)
    'mac_address': r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$',     # Validates MAC addresses
    'uuid': r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', # Validates UUIDs
    'url': r'^(https?|ftp)://[^\s/$.?#].[^\s]*$', # Validates URLs
    'email': r'^[\w\.-]+@[a-zA-Z\d\.-]+\.[a-zA-Z]{2,6}$',       # Validates email addresses
    'username': r'^[a-zA-Z0-9_]{1,32}$',                       # Validates usernames with a max length of 32 characters
    'domain': r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',               # Validates domain names
    'boolean': r'^(true|false|1|0)$',                          # Validates boolean values
    'timestamp': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$',  # Validates ISO 8601 timestamps
    'base64': r'^[A-Za-z0-9+/]+={0,2}$',                       # Validates Base64-encoded strings
    'permission_mode': r'^[0-7]{3,4}$',                        # Validates Unix permission modes (e.g., 755, 0755)
    'process_id': r'^\d+$',                                    # Validates process IDs
    'hostname': r'^[a-zA-Z0-9.-]{1,253}$',                     # Validates hostnames
}


# Add global logger
logger = logging.getLogger(__name__)

class SyscallProcessor:
    def __init__(self):
        self.syscall_frequency = Counter()
        self.argument_frequency = Counter()
        self.pattern_cache = {}
        self.suspicious_patterns = set()

    def process_syscall(self, syscall: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single syscall with enhanced validation and categorization
        """
        try:
            # Extract basic information
            syscall_name = syscall.get('name', '')
            args = syscall.get('args', [])

            # Update frequency counters
            self.syscall_frequency[syscall_name] += 1
            for arg in args:
                self.argument_frequency[str(arg)] += 1

            # Categorize syscall
            category = self._categorize_syscall(syscall_name)

            # Validate arguments
            validated_args = self._validate_arguments(args, syscall_name)

            # Check for suspicious patterns
            suspicious = self._check_suspicious_patterns(syscall_name, validated_args)

            return {
                'name': syscall_name,
                'category': category,
                'args': validated_args,
                'suspicious': suspicious,
                'timestamp': datetime.now().timestamp()
            }
        except Exception as e:
            logger.error(f"Error processing syscall: {e}")
            return syscall

    def _categorize_syscall(self, syscall_name: str) -> str:
        """
        Categorize syscall based on its function
        """
        for category, syscalls in syscall_categories.items():
            if syscall_name in syscalls:
                return category
        return 'other'

    def _validate_arguments(self, args: List[Any], syscall_name: str) -> List[str]:
        """
        Validate and normalize syscall arguments
        """
        validated_args = []
        for arg in args:
            arg_str = str(arg)

            # Check against known patterns
            arg_type = self._identify_argument_type(arg_str)

            # Normalize argument based on type
            normalized_arg = self._normalize_argument(arg_str, arg_type)

            validated_args.append(normalized_arg)

        return validated_args

    def _identify_argument_type(self, arg: str) -> str:
        """
        Identify the type of argument based on patterns
        """
        for arg_type, pattern in argument_patterns.items():
            if re.match(pattern, arg):
                return arg_type
        return 'unknown'

    def _normalize_argument(self, arg: str, arg_type: str) -> str:
        """
        Normalize argument based on its type
        """
        if arg_type == 'file_path':
            return os.path.normpath(arg)
        elif arg_type == 'number':
            return str(int(arg))
        elif arg_type == 'hex':
            return hex(int(arg, 16))
        return arg

    def _check_suspicious_patterns(self, syscall_name: str, args: List[str]) -> bool:
        """
        Check for suspicious patterns in syscall usage
        """
        pattern_key = f"{syscall_name}:{','.join(args)}"

        if pattern_key in self.pattern_cache:
            return self.pattern_cache[pattern_key]

        suspicious = False

        # Check for suspicious combinations
        if syscall_name in ['execve', 'system'] and any('/tmp/' in arg for arg in args):
            suspicious = True
        elif syscall_name in ['chmod', 'chown'] and any('777' in arg for arg in args):
            suspicious = True
        elif syscall_name in ['connect', 'bind'] and any(self._is_suspicious_port(arg) for arg in args):
            suspicious = True

        self.pattern_cache[pattern_key] = suspicious
        return suspicious

    def _is_suspicious_port(self, arg: str) -> bool:
        """
        Check if a port number is suspicious
        """
        suspicious_ports = {21, 22, 23, 3389}  # Example suspicious ports
        try:
            port = int(arg)
            return port in suspicious_ports
        except ValueError:
            return False



def convert_json_to_text(json_file: str, text_file: str, mode: str = 'test') -> bool:
    """Convert JSON log to text format with raw syscall handling"""
    processed_count = 0
    error_count = 0

    try:
        with open(json_file, 'r') as jf, open(text_file, 'w') as tf:
            for line in jf:
                try:
                    # Skip non-JSON lines
                    if not line.strip().startswith('{'):
                        continue

                    data = json.loads(line)

                    # Handle raw syscalls format
                    if 'process_tracepoint' not in data:
                        continue

                    proc_info = data['process_tracepoint'].get('process', {})
                    if 'binary' not in proc_info:
                        continue

                    # Extract syscall information from raw format
                    syscall_info = data['process_tracepoint']
                    syscall_id = syscall_info['args'][0].get('long_arg', 0)

                    # Convert syscall ID to name
                    syscall_name = syscall_map.get(str(syscall_id), f"syscall_{syscall_id}")

                    # Get argument format for this syscall
                    arg_format = syscall_arg_format.get(syscall_name, [])

                    # Extract and format arguments
                    formatted_args = []
                    if len(syscall_info['args']) > 1:
                        for i, arg in enumerate(syscall_info['args'][1:], 1):
                            # Get argument value - try both long_arg and size_arg
                            arg_value = arg.get('long_arg', arg.get('size_arg', 0))

                            # Get argument format if available
                            arg_fmt = arg_format[i - 1] if i <= len(arg_format) else None

                            if arg_fmt:
                                # Split the format into type and name
                                arg_parts = arg_fmt.split()
                                arg_type = arg_parts[0]

                                # Handle pointer types and complex types
                                if len(arg_parts) > 1:
                                    arg_name = arg_parts[-1]
                                    # Handle pointer types
                                    if '*' in arg_fmt:
                                        arg_type = f"{arg_type}(*{arg_name}"
                                    else:
                                        arg_type = f"{arg_type}({arg_name}"
                                else:
                                    arg_type = f"{arg_type}(arg{i}"

                                # Format the argument with actual value
                                formatted_args.append(f"{arg_type}={hex(int(arg_value))}")
                            else:
                                # Use generic format for unknown arguments
                                formatted_args.append(f"arg{i}={hex(int(arg_value))}")

                    # Create text line with formatted arguments
                    binary = proc_info['binary']
                    text_line = f"{binary} {syscall_name}({','.join(formatted_args)})\n"
                    tf.write(text_line)

                    processed_count += 1

                except json.JSONDecodeError as e:
                    error_count += 1
                    logging.error(f"JSON decode error: {e}")
                    continue
                except Exception as e:
                    error_count += 1
                    logging.error(f"Error processing line: {e}")
                    continue

        logging.info(f"Processed {processed_count} syscalls with {error_count} errors")
        return processed_count > 0

    except Exception as e:
        logging.error(f"Error processing file {json_file}: {e}")
        return False


def read_syscalls_from_log(log_file: str) -> List[Dict[str, Any]]:
    """Read syscalls from log with enhanced error handling and path resolution"""
    syscalls = []
    try:
        # Ensure full path resolution
        full_path = os.path.join(config.TEMP_DIR, log_file) if not os.path.isabs(log_file) else log_file

        if not os.path.exists(full_path):
            logging.error(f"Log file not found: {full_path}")
            return []

        with open(full_path, 'r') as lf:
            for line_num, line in enumerate(lf, 1):
                try:
                    # Skip empty lines
                    line = line.strip()
                    if not line:
                        continue

                    # Split binary and rest of the line
                    parts = line.split(None, 1)
                    if len(parts) < 2:
                        logging.warning(f"Invalid line format at line {line_num}: {line}")
                        continue

                    binary = parts[0]
                    rest = parts[1]

                    # Extract syscall name and arguments
                    if '(' in rest and ')' in rest:
                        syscall_name = rest[:rest.index('(')]
                        args_str = rest[rest.index('(') + 1:rest.rindex(')')]

                        # Parse structured arguments
                        args = []
                        if args_str:
                            # Split arguments by comma, but preserve commas within parentheses
                            current_arg = ""
                            paren_count = 0

                            for char in args_str:
                                if char == '(' or char == '{':
                                    paren_count += 1
                                elif char == ')' or char == '}':
                                    paren_count -= 1
                                elif char == ',' and paren_count == 0:
                                    if current_arg.strip():
                                        args.append(current_arg.strip())
                                    current_arg = ""
                                    continue
                                current_arg += char

                            if current_arg.strip():
                                args.append(current_arg.strip())

                        syscalls.append({
                            'binary': binary,
                            'name': syscall_name,
                            'args': args,
                            'timestamp': time.time(),
                            'line_number': line_num
                        })

                except Exception as e:
                    logging.error(f"Error processing line {line_num} in {full_path}: {str(e)}")
                    continue

        if syscalls:
            logging.info(f"Successfully read {len(syscalls)} syscalls from {full_path}")
        else:
            logging.warning(f"No valid syscalls found in {full_path}")

        return syscalls

    except Exception as e:
        logging.error(f"Error reading syscall log {log_file}: {str(e)}")
        return []


# Fix global variable access
def update_known_entities(syscall_name: str, arguments: List[str], mode: str = 'train') -> None:
    global known_syscalls, known_arguments, distinct_syscalls_with_unseen_args
    try:
        if mode == 'train':
            known_syscalls.add(syscall_name)
            known_arguments.update(arguments)

        if any(is_unseen_argument(arg) for arg in arguments):
            distinct_syscalls_with_unseen_args.add(syscall_name)
    except Exception as e:
        logger.error(f"Error updating known entities: {e}")


def is_unseen_syscall(syscall_name: str) -> bool:
    """
    Check if syscall is unseen with pattern validation
    """
    return syscall_name not in known_syscalls


def is_unseen_argument(argument: str) -> bool:
    """
    Check if argument is unseen with pattern validation
    """
    return argument not in known_arguments
