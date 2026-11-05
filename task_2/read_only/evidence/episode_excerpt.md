# Raw episode excerpt

These are observations and action labels from episode 0 of the original training
archive. The paddle centre is measured from the white paddle pixels in each raw
frame. Action a[t] is issued after observing frame t and produces frame t+1.
Consecutive rows let you inspect the recorded motion without loading an archive.

| Episode | Frame t | Paddle centre row | Raw action a[t] | Name |
| --- | --- | --- | --- | --- |
| 0 | 0 | 9 | 2 | DOWN |
| 0 | 1 | 11 | 2 | DOWN |
| 0 | 2 | 13 | 0 | STAY |
| 0 | 3 | 13 | 2 | DOWN |
| 0 | 4 | 15 | 0 | STAY |
| 0 | 5 | 15 | 1 | UP |
| 0 | 6 | 13 | 0 | STAY |
| 0 | 7 | 13 | 0 | STAY |
| 0 | 8 | 13 | 1 | UP |
| 0 | 9 | 11 | 2 | DOWN |
| 0 | 10 | 13 | 0 | STAY |

The contact sheet in the prompt shows dataset-loaded transitions; this table
contains the raw logged observations and actions. No model prediction is used.

Archive SHA-256: `5c03fa09977ebdc73beb6181eac96743dc21d18a219e442ee89ef89ac84f5b32`
