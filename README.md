# DLGN_Only
Separate the DLCA code from DLGN code, then make the DiffTrees for CDLGN

Why do the move? Because the code is disbursed which is driving me crazy.

We are using jax. Jax is fast, jax works well on my graphics card, and if we go there we can 'shard' it. But Vmapping is very nice, and we don't have to manage it.

DLGN

-DLGN_Types

-Datasets
--Torch_Loaders
--Python files that thresholded datasets

-DLGN_to_Verilog
--Verilog_Helper
--DWN_popcount (fix the prior attempt but with Jax)
--Librelane_code

-Jax_General
--Inits
--Runs
--helpers

This WILL NOT merge with DLCA content until after Tuesday.
