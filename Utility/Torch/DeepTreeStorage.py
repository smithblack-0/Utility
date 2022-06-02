"""

Start Root
Ready for input.
On input:
    For labels in tree. Calculate Score.
    At highest_score_node insert key.
    For node in tree:

        if node has_leaf_children_only:
            if centroid_loss(subchild_a + subchild_b) < centroid_loss(subchild_a) + centroid_losS(subchild_b) + branch_penalty:
                merge_children_and_make_leaf
        if node is leaf:
            if centroid_loss(group_a + group_b) > centroid_loss(group_a) + centroid_loss(group_b) + branch_penalty
                turn_leaf_into_branch.
            

        if node loss > create_threshold:
            #Node is actually constantly there, being trained,
            #but not contributing
            create_new_node:


        if centraid loss exceeds make_new_threshold:
            make_new_node
    For node in tree.




"""