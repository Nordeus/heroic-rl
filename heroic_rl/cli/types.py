import click


class HostPortPair(click.ParamType):

    name = "host:port"

    def convert(self, value, param, ctx):
        parts = value.split(":")
        if len(parts) != 2:
            self.fail("{} is not a valid host:port pair".format(value), param, ctx)
            return
        host = parts[0]
        if len(host) <= 0:
            self.fail("{} has empty host part".format(value), param, ctx)
            return
        try:
            _ = int(parts[1])
            return value
        except ValueError:
            self.fail("{} does not contain a valid port".format(value), param, ctx)


HOST_PORT = HostPortPair()
