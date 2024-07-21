
- ch8-p96-rnn-cell.py
Creates nn.rnnCell model and feeds points[0][0] => [1,2] that is one corner along the two coords. 
where points=[256=samples, 4=corners, 2=coordinates).
Input is [1,2] and output is [1,2] as well.

- ch8-p96-rnn-cell-modded.py
Modified version of ch8-p96-rnn-cell-full-seq.py
Mods: 
    1) Creates rnn_cell_manual additionally, a manual version of nn.rnnCell and uses it to run the same input.
    2) Outputs for nn.rnnCell (using nn library) and manual one rnn_cell_manual side by side for comparison.

- ch8-p96-rnn-cell-full-seq.py
Creates nn.rnnCell model and feeds points[0] => [4,2] where 
points=[256=samples, 4=corners, 2=coordinates) iteratively and captures the final output.
For each iteration, input is [1,2] and output is [1,2] as well.

- ch8-p96-rnn-cell-full-seq-modded.py
Modified version of ch8-p96-rnn-cell-full-seq.py
Mods: 
    1) Creates rnn_cell_manual additionally, a manual version of nn.rnnCell and uses it to run the same iteration.
    2) Outputs for nn.rnnCell (using nn library) and manual one rnn_cell_manual side by side for comparison.
 
ch8-p100-plot-data.py
Plots points, directions=>[256=samples,4=corners,2=coords], [256=directions]

- ch8-p115.py
Implements batched version with input data [3,4,2] = first three samples out of 256 rectangles.
Permutes it and then feeds into rnn model (not cell this time!) as [4,3,2]
Also feeds into rnn_cell unpermutted [3,4,2] with batch parameter so that rnn model will take care of unpermutted input.
Output of permuted and unpermutted (batch_first param) are compared [3,4,2]. 

- ch8-p115-modded.py
(has not implemented): should be implementing manual version of rnn.

- ch8-p131-bidir-rnn.py
Implements bidirection version of rnn.

- ch8-p134-full-classificiation-model-modded.py
Implements full prediction using rnn model and points,directions data.o
1) implements square model for this which encapsulates nn.Rnn with full features: forward, internal data etc.,
2) section points direction into training and test/validation data.
3) both during training and testing/validation, sections data into mini_batches along with epochs. 
4) accumulates training losses and test/validation losses during each epoch (100 times) and graphs it as comparison.

ch8-p134-full-classificiation-model.py
ch8.py
