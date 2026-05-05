We use the confidence value of the depthmaps with pairwise predictions.

Pairwise predictions are as follows.
Let q be the query image and r_1, r_2, ..., r_5 the top 5 references estimated by salad.

We use VGGT with 5 sequences of len 2 pairing each ref image agianst the query:

VGGT({q, r1})
VGGT({q, r2})
....
VGGT({q, r5})

Then we take the confidence maps and compute a metric such as:
- The sum
- The sum of the top25 values

And take these values to rerank {r1, r2, ..., r5}

This indeed does not work.
Using confidence value may be a really bad idea since
we can't explain the magnitude of the items of the confidence map.

Probably a full prediction could make sense?


The following logs are used as reference

----------------------------------------------------------------------------------------------------
95it [01:38,  1.11s/it]tensor([ 95,  47, 595,  93, 235]) [95]
tensor([235,  47,  93, 595,  95]) tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
VGGT FAILED
----------------------------------------------------------------------------------------------------
96it [01:39,  1.12s/it]tensor([ 96, 101, 466,  97,  56]) [96]
tensor([ 96, 466, 101,  97,  56]) tensor([3.0445, 2.0470, 2.9937, 1.4111, 1.0000])
----------------------------------------------------------------------------------------------------
97it [01:40,  1.12s/it]tensor([ 97,   2, 114, 579,   1]) [97]
tensor([114,   1, 579,   2,  97]) tensor([1.0000, 1.0000, 2.0309, 1.0000, 1.0037])
VGGT FAILED
----------------------------------------------------------------------------------------------------
98it [01:41,  1.09s/it]tensor([ 98,  99, 366, 108, 459]) [98]
tensor([ 98,  99, 108, 459, 366]) tensor([4.9843, 3.3620, 1.0026, 1.1541, 1.0759])
----------------------------------------------------------------------------------------------------
99it [01:42,  1.10s/it]tensor([ 99,  98, 108, 459, 324]) [99]
tensor([ 98,  99, 108, 459, 324]) tensor([1.6963, 3.3879, 1.0394, 1.0000, 1.0000])
VGGT FAILED
----------------------------------------------------------------------------------------------------
100it [01:44,  1.04s/it]
92 62 torch.Size([100, 8448]
