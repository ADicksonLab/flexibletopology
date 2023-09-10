proc load_attributes {filename} {
    set f [open $filename]
    set t [read $f]
    close $f

    set attr_all [split $t "\n"]

    # attributes will be accessible in the main function as an array
    # stored as "attr_variable"
    global ft_attr
    global ft_idxs

    set n_atoms [llength $ft_idxs]
    puts "num atoms: $n_atoms"
    
    set at 0
    set frame -1
    set i 0
    foreach attr_line $attr_all {
	# remove newline character
	set attr_line [string trim $attr_line]

	# determine correct atom, frame index
	if {[expr $i % $n_atoms] == 0} {
	    incr frame
	    set at 0
	}

	# split attr_line by space
	set attrs [split $attr_line " "]

	# save attrs to multidim array
	for {set a 0} {$a < 4} {incr a} {
	    set ft_attr($at,$frame,$a) [lindex $attrs $a]
	}

	# increment counters
	incr i
	incr at
    }
    return
}

proc delete_reps {mol} {
    # deletes all existing representations
    set n [molinfo $mol get numreps]
    for {set rep 0} {$rep < $n} {incr rep} {
	mol delrep 0 $mol
    }
}

proc range {from to {step 1}} {
     set res $from
     while {$step>0?$to>$from:$to<$from} {lappend res [incr from $step]}
     return $res
}

proc setup_mol_reps {mol} {
    # prepares the molecular representations for molecule "mol"
    # using a separate VDW rep for each atom in the list "idxs"

    global ft_idxs
    global ft_mode
    global ft_firstrep

    set gh_sel [atomselect $mol "resname gho"]
    set ft_idxs [$gh_sel list]
    
    set ft_mode charge

    set ft_firstrep [molinfo $mol get numreps]
    
    foreach atom $ft_idxs {
	mol representation VDW 0.5 27.000000
	mol selection index $atom
	mol material Opaque
	mol color Beta
	set idx [mol addrep $mol]
	mol colupdate $idx $mol 1
	#mol scaleminmax $mol $idx 0.000000 1.000000
    }
}

proc set_ft_mode {mode} {
    global ft_mode
    
    if {$mode == "charge"} {
	set ft_mode charge
    } elseif {$mode == "epsilon"} {
	set ft_mode epsilon
    } else {
	puts "Unknown mode: $mode"
    }
}
	
    

proc set_attr_state {mol frame} {
    # sets the visualization state for molecule "mol"
    # to the data stored in the "atomic_attributes" array for frame "frame"
    # the "mode" can either be set to "charge" (default) or "epsilon"
    global ft_attr
    global ft_idxs
    global ft_mode
    global ft_firstrep
    
    set natoms [llength $ft_idxs]
    set size [array size ft_attr]
    set nframes [expr $size/$natoms]

    if {$frame >= $nframes} {
	puts "Error! There are only $nframes frames and you asked for frame $frame!"
	return
    } else {
	for {set i 0} {$i < $natoms} {incr i} {
	    set charge $ft_attr($i,$frame,0)
	    set sigma $ft_attr($i,$frame,1)
	    set epsilon $ft_attr($i,$frame,2)
	    set lambda $ft_attr($i,$frame,3)

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
	    set idx [lindex $ft_idxs $i]
	    set sel [atomselect $mol "index $idx"]
	    
	    $sel set beta $colorind
	    
	    mol modcolor $rep_idx $mol Beta
	    mol scaleminmax $mol $rep_idx 0.25 0.75000
	    
	    
	    # set the size using sigma
	    set sizemin 0.2
	    set sizemax 0.7
	    set sigmin 0.13
	    set sigmax 0.5
	    set sigfrac [expr ($sigma - $sigmin)/($sigmax - $sigmin)]
	    if {$sigfrac < 0} {
		set sigfrac 0
	    } elseif {$sigfrac > 1} {
		set sigfrac 1
	    }
	    set size [expr $sizemin + $sigfrac*($sizemax-$sizemin)]
	    mol modstyle $rep_idx $mol VDW $size 27.000000

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
    set_attr_state $mol $frame
    return
}

proc ft_step {step {mol 0}} {
    set f [molinfo $mol get frame]
    set f2 [expr $f + $step]
    animate goto $f2
    set_attr_state $mol $f2
    return
}

proc ft_update {{mol 0}} {
    set f [molinfo $mol get frame]
    set_attr_state $mol $f
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
