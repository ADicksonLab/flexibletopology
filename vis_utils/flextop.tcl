proc ft_load_attributes {mol filename} {
    set f [open $filename]
    set t [read $f]
    close $f

    set attr_all [split $t "\n"]

    # attributes will be accessible in the main function as an array
    # stored as "attr_variable"
    global ft_attr
    global ft_idxs
    global ft_n_frames
    
    set n_atoms [llength $ft_idxs($mol)]
    puts "num atoms: $n_atoms"
    
    set at 0
    set frame -1
    set i 0
    set ft_n_frames($mol) 0

    foreach attr_line $attr_all {
	# remove newline character
	set attr_line [string trim $attr_line]

	# determine correct atom, frame index
	if {[expr $i % $n_atoms] == 0} {
	    incr frame
	    incr ft_n_frames($mol)
	    set at 0
	}

	# split attr_line by space
	set attrs [split $attr_line " "]

	# save attrs to multidim array
	for {set a 0} {$a < 4} {incr a} {
	    set ft_attr($mol,$at,$frame,$a) [lindex $attrs $a]
	}

	# increment counters
	incr i
	incr at
    }
    return
}

proc ft_delete_reps {mol} {
    # deletes all existing representations
    set n [molinfo $mol get numreps]
    for {set rep 0} {$rep < $n} {incr rep} {
	mol delrep 0 $mol
    }
}

proc ft_setup_mol_reps {mol} {
    # prepares the molecular representations for molecule "mol"
    # using a separate VDW rep for each atom in the list "idxs"

    global ft_idxs
    global ft_mode
    global ft_firstrep

    set gh_sel [atomselect $mol "resname gho"]
    set ft_idxs($mol) [$gh_sel list]
    
    set ft_mode charge

    set ft_firstrep [molinfo $mol get numreps]
    
    foreach atom $ft_idxs($mol) {
	mol representation VDW 0.5 27.000000
	mol selection index $atom
	mol material Opaque
	mol color Beta
	set idx [mol addrep $mol]
	mol colupdate $idx $mol 1
	#mol scaleminmax $mol $idx 0.000000 1.000000
    }
}

proc ft_set_mode {mode} {
    global ft_mode
    
    if {$mode == "charge"} {
	set ft_mode charge
    } elseif {$mode == "epsilon"} {
	set ft_mode epsilon
    } else {
	puts "Unknown mode: $mode"
    }
}
	
    

proc ft_set_attr_state {mol frame} {
    # sets the visualization state for molecule "mol"
    # to the data stored in the "atomic_attributes" array for frame "frame"
    # the "mode" can either be set to "charge" (default) or "epsilon"
    global ft_attr
    global ft_idxs
    global ft_mode
    global ft_firstrep
    global ft_n_frames
    
    set natoms [llength $ft_idxs($mol)]

    if {$frame >= $ft_n_frames($mol)} {
	puts "Error! There are only $ft_n_frames($mol) frames and you asked for frame $frame!"
	return
    } else {
	for {set i 0} {$i < $natoms} {incr i} {
	    set charge $ft_attr($mol,$i,$frame,0)
	    set sigma $ft_attr($mol,$i,$frame,1)
	    set epsilon $ft_attr($mol,$i,$frame,2)
	    set lambda $ft_attr($mol,$i,$frame,3)
	    #puts "charge: $charge"
	    #puts "sigma: $sigma"
	    #puts "epsilon: $epsilon"
	    #puts "lambda: $lambda"

	    # set the color using either charge or epsilon
	    if {$ft_mode == "charge"} {
		set cmin -1
		set cmax 1
		set colorind [expr ($charge - $cmin)/($cmax-$cmin)]
		if {$colorind < 0} {
		    set colorind 0
		} elseif {$colorind > 1} {
		    set colorind 1
		}
	    } elseif {$ft_mode == "epsilon"} {
		set cmin 0.2
		set cmax 2
		set colorind [expr ($epsilon - $cmin)/($cmax-$cmin)]
	    }

	    # use the ft_firstrep offset to determine the index of the representation to alter
	    set rep_idx [expr $i + $ft_firstrep]

	    # get i-th ghost atom index
	    set idx [lindex $ft_idxs($mol) $i]
	    set sel [atomselect $mol "index $idx"]
	    
	    $sel set beta $colorind
	    
	    mol modcolor $rep_idx $mol Beta
	    mol scaleminmax $mol $rep_idx 0.25 0.75000
	    
	    # set the sigscale using sigma (as a fraction of the Argon sigma radius: 0.34 nm)
	    set sigscale [expr $sigma/0.34]
	    mol modstyle $rep_idx $mol VDW $sigscale 27.000000

	    # use lambda to determine whether to render in opaque (lambda > 0.7), transparent (0.3 < lambda < 0.7), or ghost (lambda < 0.3) 
	    if {$lambda > 0.7} {
		mol modmaterial $rep_idx $mol Opaque
	    } elseif {$lambda > 0.3} {
		mol modmaterial $rep_idx $mol Transparent
	    } else {
		mol modmaterial $rep_idx $mol Ghost
	    }
	}
    }   
}
	  
proc ft_goto {frame {mol 0}} {
    animate goto $frame
    ft_set_attr_state $mol $frame
    return
}

proc ft_step {step {mol 0}} {
    set f [molinfo $mol get frame]
    set f2 [expr $f + $step]
    animate goto $f2
    ft_set_attr_state $mol $f2
    return
}

proc ft_update {{mol 0}} {
    set f [molinfo $mol get frame]
    ft_set_attr_state $mol $f
    return
}    

proc ft_load_heating {} {
    set mol [mol new struct_before_min.pdb]
    mol addfile minimized_pos.pdb $mol
    mol addfile heating0.dcd -waitfor -1 $mol
    mol	addfile	heating1.dcd -waitfor -1 $mol
    mol	addfile	heating2.dcd -waitfor -1 $mol
    mol	addfile	heating3.dcd -waitfor -1 $mol
    mol	addfile	heating4.dcd -waitfor -1 $mol
    mol	addfile	heating5.dcd -waitfor -1 $mol
    mol	addfile	heating6.dcd -waitfor -1 $mol
    mol	addfile	heating7.dcd -waitfor -1 $mol
    return
}
