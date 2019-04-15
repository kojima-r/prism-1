$pp_index_get_i1(0,[],_).
$pp_index_get_i1(N0,[X|G],S):-N0>0,N is N0 -1,index_atoms(As),member(X,As),not member(X,S),$pp_index_get_i1(N,G,[X|S]).
$pp_index_all_different(N,G):-findall(X,$pp_index_get_i1(N,X,[]),G).
$pp_index_get_i2(0,[],_).
$pp_index_get_i2(N0,[X|G],S):-N0>0,N is N0 -1,index_atoms(As),member(X,As),$pp_index_get_i2(N,G,[X|S]).
$pp_index_all_combination(N,G):-findall(X,$pp_index_get_i2(N,X,[]),G).

%%%%
%%%% utility
%%%%

set_index_range(Index,R):-
	$pc_set_index_range(Index,R).

save_placeholder_goals(PlaceholderGoals,Goals):-save_placeholder_goals('data.h5',hdf5,PlaceholderGoals,Goals).
save_placeholder_goals(Filename,PlaceholderGoals,Goals):-save_placeholder_goals(Filename,hdf5,PlaceholderGoals,Goals).
save_placeholder_goals(Filename,Mode,PlaceholderGoals,Goals):-
	$pp_generate_placeholder_goals(PlaceholderGoals,Goals,0,GG,Var),
	save_placeholder_data(Filename,Mode,Var,GG).

$pp_generate_placeholder([],N,N).
$pp_generate_placeholder([V|VV],N,O):-
	M is N+1,
	term2atom(M,AM),
	atom_concat($placeholder,AM,V0),
	atom_concat(V0,$,V),
	$pp_generate_placeholder(VV,M,O).

$pp_generate_placeholder_goals([],_,_,[],[]).
$pp_generate_placeholder_goals([GP0|GP],Goals,N,[X|XX],[V0|Var]):-
	copy_term(GP0,GP1),
	term_variables(GP0,V0),
	term_variables(GP1,V1),
	$pp_generate_placeholder(V0,N,M),
	findall(V1,member(GP1,Goals),X),
	$pp_generate_placeholder_goals(GP,Goals,M,XX,Var).
	

save_placeholder_data(Placeholders,Data):-save_placeholder_data('data.h5',hdf5,Placeholders,Data).
save_placeholder_data(Filename,Placeholders,Data):-save_placeholder_data(Filename,hdf5,Placeholders,Data).
save_placeholder_data(Filename,Mode,Placeholders,Data):-
	(Mode==json ->Mode0=0
	;Mode==pb   ->Mode0=1
	;Mode==pbtxt->Mode0=2
	;Mode==hdf5 ->Mode0=3
	;$pp_raise_runtime_error($msg(9804),unknown_save_format,save_placeholder_data/4)),
	$pc_save_placeholder_data(Filename,Mode0,Placeholders,Data,20000000).

%%%
%%% save prism flags
%%%

save_flags:-save_flags('flags.json',json).
save_flags(Filename):-save_flags(Filename,json).
save_flags(Filename,Mode,[]):-save_flags(Filename,Mode,[]).
save_flags(Filename,Mode,TensorList):-
	findall([X,F],get_prism_flag(X,F),G),
	$pc_set_export_flags(G),
	(Mode==json ->Mode0=0
	;Mode==pb   ->Mode0=1
	;Mode==pbtxt->Mode0=2
	;Mode==hdf5 ->Mode0=$pp_raise_runtime_error($msg(9806),hdf5_is_not_supportted_for_saving_flags,save_flags/2)
	;$pp_raise_runtime_error($msg(9804),unknown_save_format,save_flags/2)),
	$pc_save_options(Filename,Mode0,TensorList).

%%%%
%%%% save explanation graph
%%%%

unique([],[]).
unique([X],[X]).
unique([H|T],[H|S]) :- not(member(H, T)), unique(T,S).
unique([H|T],S) :- member(H, T), unique(T,S).

$pp_find_unifiable(X,X,[],[]).
$pp_find_unifiable(X,Z,[Y|L],L1):-X?=Y,subsumes_term(X,Y)->$pp_find_unifiable(X,Z,L,L1);$pp_find_unifiable(Y,Z,L,L1).
$pp_find_unifiable(X,Z,[Y|L],[Y|L1]):-not X?=Y,$pp_find_unifiable(X,Z,L,L1).
unifiable_unique([],[]).
unifiable_unique([H|T0],[Z|S]) :- $pp_find_unifiable(H,Z,T0,T1),unifiable_unique(T1,S).


$pp_tensor_filter_nonground([],[]).
$pp_tensor_filter_nonground([H|T],[H|S]) :- ground(H), unique(T,S).
$pp_tensor_filter_nonground([H|T],S) :- not(ground(H)), unique(T,S).

nonground_unique(L,L1):-
	$pp_tensor_filter_nonground(L,L0),
	unique(L0,L1).

%CollectLists:
% data(occured switches,declared index)
$pp_trans_phase_tensor(Prog0,Prog_tensor,Info):-
	format(">>=====\n"),
	maplist(X,Y,CollectList,(
		X=pred(index,2,A2,A3,A4,Clauses)->
			($pp_tensor_parse_clauses(Clauses,C,CollectList),
			Y=pred(values,2,A2,A3,A4,C));
		X=pred(A0,A1,A2,A3,A4,Clauses)->
			($pp_tensor_parse_clauses(Clauses,C,CollectList),
			Y=pred(A0,A1,A2,A3,A4,C));
		Y=X,CollectList=[]
	),Prog0,Prog_tensor0,CollectLists),
	format(">>=====\n"),
	flatten(CollectLists,FlatCList),
	%=====
	% declared_index_atoms:
	% occured_index_atoms:
	% index_atoms:
	%
	maplist(C,X,C=data(_,X,_,_),FlatCList,IndexList),
	flatten(IndexList,IndexAtoms),
	nonground_unique(IndexAtoms,UIndexAtoms),
	assert(declared_index_atoms(UIndexAtoms)),
	%
	maplist(C,X,C=data(X,_,_,_),FlatCList,TensorList),
	flatten(TensorList,TensorAtoms),
	maplist(T,I,T=tensor(_,I),TensorAtoms,TIndexList),
	flatten(TIndexList,TIndexAtoms),
	nonground_unique(TIndexAtoms,TUIndexAtoms),
	assert(occuered_index_atoms(TUIndexAtoms)),
	%
	append(UIndexAtoms,TUIndexAtoms,AllIndexAtoms),
	nonground_unique(AllIndexAtoms,UAllIndexAtoms),
	assert(index_atoms(UAllIndexAtoms)),
	%========
	% add operator
	maplist(C,X,C=data(_,_,X,_),FlatCList,TAList),
	flatten(TAList,TAL),
	maplist(TA,Val,(copy_term(TA,TA1),Val=(values(tensor(TA1),G):-tensor_atom(TA1,Shape),length(Shape,N),$pp_index_all_combination(N,G))),TAL,ValPreds),
	Pred2=pred(values,2,_,_,_,ValPreds),
	Pred1=pred(values,3,_,_,_,[values($operator(_),[$operator],fix@[1.0])]),
	%========
	% add subgoal
	maplist(C,X,C=data(_,_,_,X),FlatCList,SGList),
	flatten(SGList,SG0List),
	maplist(SG,X,(SG=subgoal(G,_),G=..[F|Args],length(Args,L),X=F/L),SG0List,SG1List),
	nonground_unique(SG1List,SG2List),
	maplist(G,Pred,(G=PredName/NArg,length(Arg,NArg),A=..[PredName|Arg],Pred=(subgoal(A,S):-msw($operator(reindex(S)),$operator),A)),SG2List,SGPreds),
	Pred3=pred(subgoal,2,_,_,_,SGPreds),
	%========
	$pp_tensor_merge_pred(Prog_tensor0,Pred1,Prog_tensor1),
	$pp_tensor_merge_pred(Prog_tensor1,Pred2,Prog_tensor2),
	$pp_tensor_merge_pred(Prog_tensor2,Pred3,Prog_tensor),
	format(">> T-PRISM before\n"),
	maplist(X,format("~w\n",X),Prog0),
	format(">> T-PRISM after\n"),
	maplist(X,format("~w\n",X),Prog_tensor),
	format("=====\n").
$pp_tensor_merge_pred([],Pred,NewProg):-NewProg=[Pred].
$pp_tensor_merge_pred([pred(G,A,A1,A2,A3,Cl)|Rest],pred(G,A,_,_,_,ClPred),NewProg):-
	append(Cl,ClPred,NewCl),
	[pred(G,A,A1,A2,A3,NewCl)|Rest]=NewProg.
$pp_tensor_merge_pred([Prog|Rest],Pred,[Prog|NewProg]):-
	$pp_tensor_merge_pred(Rest,Pred,NewProg).
	
$pp_tensor_parse_clauses(Clauses,NewClauses,CollectLists):-
	maplist(C,NC,CollectList,(
		$pp_tensor_get_msws(C,NC,CollectList)
		),Clauses,NewClauses,CollectLists).

$pp_tensor_get_msws(Clause,MswClause,CollectList):-
	Clause=tensor_atom(_,_) -> (
		Clause=tensor_atom(TA,_),
		CollectList=data([],[],TA,[]),
		MswClause=Clause);
	Clause=index(_,_) -> (
		Clause=index(A0,A1),
		CollectList=data([],A1,[],[]),
		MswClause=values(tensor(A0),A1));
	Clause=(index(_,_):-Body) -> (
		Clause=(index(A0,A1):-Body),
		CollectList=data([],[],[],[]),
		MswClause=(values(tensor(A0),A1):-Body));
	Clause=(H:-Body) -> (
		Clause=(H:-Body),
		and_to_list(Body,LBody),
		$pp_tensor_msw_filter(LBody,MswLBody,VecList),
		$pp_tensor_subgoal_filter(LBody,Subgoals),
		list_to_and(MswLBody,MswBody),
		CollectList=data(VecList,[],[],Subgoals),
		MswClause=(H:-(MswBody)));
	MswClause=Clause,CollectList=data([],[],[],[]).

$pp_tensor_subgoal_filter([],[]).
$pp_tensor_subgoal_filter([subgoal(A0,A1)|Atoms],[subgoal(A0,A1)|SGList]):-$pp_tensor_subgoal_filter(Atoms,SGList).
$pp_tensor_subgoal_filter([_|Atoms],SGList):-$pp_tensor_subgoal_filter(Atoms,SGList).


$pp_tensor_msw_filter([],[],[]).
$pp_tensor_msw_filter([tensor(A0,A1)|Atoms],[msw(tensor(A0),A1)|Msws],[tensor(A0,A1)|VecList]):-$pp_tensor_msw_filter(Atoms,Msws,VecList).
$pp_tensor_msw_filter([vector(A0,A1)|Atoms],[msw(tensor(A0),A1)|Msws],[tensor(A0,A1)|VecList]):-$pp_tensor_msw_filter(Atoms,Msws,VecList).
$pp_tensor_msw_filter([matrix(A0,A1)|Atoms],[msw(tensor(A0),A1)|Msws],[tensor(A0,A1)|VecList]):-$pp_tensor_msw_filter(Atoms,Msws,VecList).
$pp_tensor_msw_filter([operator(A0)|Atoms],[msw($operator(A0),$operator)|Msws],VecList):-$pp_tensor_msw_filter(Atoms,Msws,VecList).
$pp_tensor_msw_filter([A|Atoms],[A|Msws],VecList):-$pp_tensor_msw_filter(Atoms,Msws,VecList).

$pp_tensor_values_filter([],[],[]).
$pp_tensor_values_filter([index(A0,A1)|Atoms],[values(tensortor(A0),A1)|Msws],[index(A0,A1)|VecList]):-$pp_tensor_values_filter(Atoms,Msws,VecList).
$pp_tensor_values_filter([A|Atoms],[A|Msws],VecList):-$pp_tensor_values_filter(Atoms,Msws,VecList).


%%%%%%%%%%%

$pp_tensor_value_generator([],V,V).
$pp_tensor_value_generator([Vec|VecLists],InValues,OutValues):-
	$pp_tensor_value_generator1(Vec,InValues,Values),
	$pp_tensor_value_generator(VecLists,Values,OutValues).

$pp_tensor_value_generator1([],V,V).
$pp_tensor_value_generator1([Vec|VecLists],InValues,OutValues):-
	$pp_tensor_value_generator2(Vec,InValues,Values),
	$pp_tensor_value_generator1(VecLists,Values,OutValues).


$pp_tensor_value_generator2([],V,V).
$pp_tensor_value_generator2([Vec|VecLists],InValues,OutValues):-
	$pp_tensor_conv_tensorterm(Vec,NewVec),
	$pp_tensor_conv_valterm(NewVec,NewVal),
	%InValues,Values
	Values=[NewVal | InValues],
	$pp_tensor_value_generator2(VecLists,Values,OutValues).

$pp_tensor_conv_tensorterm(Pred,NewPred):-
	Pred=tensor(A1term,A2term),
	%A1term=..[F|Args],
	%$pp_tensor_conv_args(Args,NewArgs),
	%A1newterm=..[F|NewArgs],
	%%
	term2atom(A2term,A2atom),
	atom_concat('dummy_',A2atom,A2newatom),
	NewPred=tensor(A1term,A2newatom).
	
$pp_tensor_conv_valterm(Pred,NewPred):-
	Pred=tensor(A1term,A2term),
	NewPred=pred(values,2,_,_,_,[values(A1term,[A2term])]).

$pp_tensor_conv_args([],[]).
$pp_tensor_conv_args([Arg|Args],[NewArg|NewArgs]):-
	nonvar(Arg)->Arg=NewArg;(
		term2atom(Arg,A1atom),
		atom_concat('dummy_',A1atom,NewArg)),
	$pp_tensor_conv_args(Args,NewArgs).

%%%%%%%%%%%%

save_expl_graph:-
	$pp_learn_data_file(DataName),
	load_clauses(DataName,Goals,[]),!,
	save_expl_graph(Goals).
save_expl_graph(Goals) :-$pp_tensor_core('expl.json','flags.json',json,json,Goals).
save_expl_graph(Filename,OptionFilename) :-
	$pp_learn_data_file(DataName),
	load_clauses(DataName,Goals,[]),!,
	save_expl_graph(Filename,OptionFilename,Goals).
save_expl_graph(Filename,OptionFilename,Goals) :-$pp_tensor_core(Filename,OptionFilename,json,json,Goals).
save_expl_graph(Filename,OptionFilename,Mode,OptionMode,Goals) :-$pp_tensor_core(Filename,OptionFilename,Mode,OptionMode,Goals).

$pp_tensor_core(Filename,OptionFilename,Mode,OptionMode,GoalList) :-
	flatten(GoalList,Goals),
	$pp_learn_check_goals(Goals),
	$pp_learn_message(MsgS,MsgE,MsgT,MsgM),
	cputime(Start),
	$pp_clean_learn_info,
	$pp_trans_goals(Goals,GoalCountPairs,AllGoals),!,
	global_set($pg_observed_facts,GoalCountPairs),
	cputime(StartExpl),
	global_set($pg_num_goals,0),
	$pp_find_explanations(AllGoals),!,
	$pp_print_num_goals(MsgS),
	cputime(EndExpl),
	$pp_format_if(MsgM,"Exporting switch information ... "),
	flush_output,
	$pp_export_sw_info,
	$pp_format_if(MsgM,"done~n"),
	$pp_observed_facts(GoalCountPairs,GidCountPairs,
					   0,Len,0,NGoals,-1,FailRootIndex),
	$pc_clear_goal_rank,
	$pc_set_goal_rank(GoalList),
	$pc_prism_prepare(GidCountPairs,Len,NGoals,FailRootIndex),
	cputime(StartEM),
	%$pp_em(0,Output),
	%format("=====***==================\n"),
	(Mode==json ->Mode0=0
	;Mode==pb   ->Mode0=1
	;Mode==pbtxt->Mode0=2
	;Mode==hdf5 ->Mode0=$pp_raise_runtime_error($msg(9806),hdf5_is_not_supportted_for_saving_flags,save_flags/2)
	;$pp_raise_runtime_error($msg(9804),unknown_save_format,save_flags/2)),
	$pc_prism_save_expl_graph(Filename,Mode0,NewSws),
	format(">>>>~w\n",NewSws),
	maplist(S,SwName,(S=sw(_,SwIns),$pp_decode_switch_name(SwIns,SwName)),NewSws,Sws),
	format(">>>>~w\n",Sws),
	filter(tensor(_),Sws,Sws1),
	maplist(S,Shape,(S=tensor(X)->(tensor_atom(X,Sh),Shape=[X,Sh]);Shape=unknown),Sws1,SwShape),
	cputime(EndEM),
	save_flags(OptionFilename,OptionMode,SwShape),
	cputime(End),!.
	%$pc_import_occ_switches(NewSws,NSwitches,NSwVals),
	%format(">>>>~w\n",NewSws),
	%format("=======================\n"),
	%$pp_decode_update_switches(Mode,NewSws),
	%format("======================\n"),
	%$pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
	%$pp_assert_learn_stats(Mode,Output,NSwitches,NSwVals,TableSpace,
	%					   Start,End,StartExpl,EndExpl,StartEM,EndEM,1000),
	%$pp_print_learn_stats_message(MsgT),!.


