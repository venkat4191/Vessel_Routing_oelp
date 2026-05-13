"""
Local API key ring for LinerNet.

All 10 Gemini keys for round-robin rotation.
Keys are used one at a time. When a key hits its rate limit (429),
the pipeline automatically moves to the next key.
"""

API_KEYS = [
    "AIzaSyAPlIQDKXb-WOoaipwXN3DFaZ58XnU0WFY",
    "AIzaSyC1JXIY8doddGk3Q8fRIra7hbDW3lr7Dwc",
    "AIzaSyCN7A3WPJvZMQTH5OfluvswGWxKAEgxNao",
    "AIzaSyDtR9Ie9aWKeoexoEOu5TpEpaDFUZNxfEw",
    "AIzaSyC6AJccRmHE9deVVtW2wUHkq_PHe5hcM54",
    "AIzaSyAhCRfyYfWUoI9xPj0-3zTEYwE6JV1xuMY",
    "AIzaSyBl7xRg_Ms-LPChxXVd2MY7Pb6ZHI4O_To",
    "AIzaSyDGec9QXjrFiBEWOiWOKNjSgzL5_fqjfPg",
    "AIzaSyD--doNtlZJ7KMyN25xy9T4Ql5gb3vioxw",
    "AIzaSyAykX2oiF7esj4IfXztjU1JpVnlO6doMOo",
    "AIzaSyAXmCJh92lZ9WljWXX874PpPCvmKR3rkgg",
    "AIzaSyAlcNQrs3dVonM1KCW4IZrCSQ0flrGfgNs",
    "AIzaSyDURDz3RdbXZHVzsBsDKCC4KS1NsEN_2Rc",
    "AIzaSyD__M9DOTFzeR2tMA_-we5mlhdiUx_Ec1o",
    "AIzaSyAzB7oHcQ7y-dvrB8zo0gum9T-1Z_l_o6I",
    "AIzaSyBzM5Tb4v_QPgpTYO46G7AxYy33U2jbL9k",
    "AIzaSyD1j_6GuSxmKEKcQbgYl6YDYFwAidMTedQ",
    "AIzaSyA6AUWJmjtVEKyqumSSISeeKVdOSiX58Io",
    "AIzaSyAn_U_dPy6YcoGRr5DEr0Ri7R3Ipt80MCM",
    "AIzaSyAaktXTuy3zGAapQLjGB0tWo4LeYjZpA9o",
    "AIzaSyCyD2LXct82KDzeeFqFVbyzkrQhMLGilus",
    "AIzaSyAmMCdQ9eMQzexlJ4zCYdKzqqWkdVJuBUM",
    "AIzaSyA9DPiHkqiEuUoyqt6tnmTLOPlldZ5-0II",
    "AIzaSyC10ViUKKST15ngBDHiUL1memUoGF3_ZIQ",
    "AIzaSyD5nRDHypvrVrCooAB61K-PB-sPMILKt2E",
    "AIzaSyBsuurJ2v4U56R2Bzq54jThrOdnGDsHYmo",
    "AIzaSyBC53L313jj1261guFXBI0B-2idtkaOPvY",
    "AIzaSyBF7em1L6DJQD7L-N36vVNguv2t9sq3KP4",
    "AIzaSyCNmt4MsJynwJYBxCrc24dCfAOAO6hU56s",
    "AIzaSyCCJdoHbL82d5XnAhed3yyMWV5eR5L8f6U",
    "AIzaSyAFovpV4NpnGWFTpqpNEWd0PhDLNxVeC3Y",
    "AIzaSyDWXCTAruxiKYlKM-NfSxA5h7eJwZSvnss",
    "AIzaSyDXEoaHWTdqn7VJ3j50_CFtJY7knjEkn70",
    "AIzaSyCF9Ssn_p0IId5tyJzot4TeJbk6u-8P-z0",
    "AIzaSyA0LppTrD5xfxdoJ4TCUA80Elz4U0bqjnE",
    "AIzaSyA6P_i3fk566Aqtfdj2SeckdbGQORxvUKc",
    "AIzaSyD9NPZIE2y1YzUXDzcOjO_xGAAflxg3_JI",
    "AIzaSyA-baR5a2rDCE0p6R8jrqeugIgl_Rqm4dk",
    "AIzaSyDQpVnW36GnVmu2nwCIsoAffoKafszUaqA"
]
