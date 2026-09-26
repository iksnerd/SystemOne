# Security

Please report a vulnerability privately, through GitHub's private vulnerability reporting on this
repository (Security tab, "Report a vulnerability"), not in a public issue.

verdict's server binds to localhost and has no authentication, by design: it is not meant to be
reachable from another machine. A report that it can be made reachable without changing that
default is in scope; running it on a public interface on purpose is not.
