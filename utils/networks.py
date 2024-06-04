import network_diffusion as nd


def get_aucs_network(file_path: str) -> nd.mln.MultilayerNetwork:
    net = nd.MultilayerNetwork.from_mpx(file_path)
    net.layers.pop('coauthor')
    net.layers.pop('work')
    net.layers.pop('leisure')
    net.layers["contagion"] = net.layers.pop("lunch")
    net.layers["awareness"] = net.layers.pop("facebook")
    return net.to_multiplex()
