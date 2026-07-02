Addressee and Response Selection for Multi-Party Conversation
|                                     |     | HirokiOuchi |     |     |     |                   | YutaTsuboi |     |     |     |     |
| ----------------------------------- | --- | ----------- | --- | --- | --- | ----------------- | ---------- | --- | --- | --- | --- |
| NaraInstituteofScienceandTechnology |     |             |     |     |     | IBMResearch-Tokyo |            |     |     |     |     |
| ouchi.hiroki.nt6@is.naist.jp        |     |             |     |     |     | yutat@jp.ibm.com  |            |     |     |     |     |
Abstract
| To create                          | conversational |               | systems      | working in |     |     |     |     |     |     |     |
| ---------------------------------- | -------------- | ------------- | ------------ | ---------- | --- | --- | --- | --- | --- | --- | --- |
| actual situations,                 |                | it is crucial | to assume    | that       |     |     |     |     |     |     |     |
| they interact                      | with           | multiple      | agents.      | In this    |     |     |     |     |     |     |     |
| work, we                           | tackle         | addressee     | and response | se-        |     |     |     |     |     |     |     |
| lectionformulti-partyconversation, |                |               |              | inwhich    |     |     |     |     |     |     |     |
systemsareexpectedtoselectwhomtheyad-
| dressaswellaswhat |     | theysay. | Thekeychal- |     |     |     |     |     |     |     |     |
| ----------------- | --- | -------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
lenge of this task is to jointly model who is Figure 1: Addressee and response selection for multi-party
| talkingaboutwhat |           | inapreviouscontext.      |             | For    |               |          |               |             |                    |     |             |
| ---------------- | --------- | ------------------------ | ----------- | ------ | ------------- | -------- | ------------- | ----------- | ------------------ | --- | ----------- |
|                  |           |                          |             |        | conversation. | A        | SYSTEM        | is required | to select          | an  | appropriate |
| the joint        | modeling, | we                       | propose two | model- |               |          |               |             |                    |     |             |
|                  |           |                          |             |        | addressee     | from the | interlocutors | in          | the conversational |     | context     |
| ingframeworks:   |           | 1)staticmodelingand2)dy- |             |        |               |          |               |             |                    |     |             |
andanappropriateresponsefromthefixedsetofcandidates.
| namic modeling.    |         | To show | benchmark   | results     |            |           |     |           |          |     |           |
| ------------------ | ------- | ------- | ----------- | ----------- | ---------- | --------- | --- | --------- | -------- | --- | --------- |
| of our frameworks, |         | we      | created a   | multi-party |            |           |     |           |          |     |           |
| conversation       | corpus. | Our     | experiments | on the      |            |           |     |           |          |     |           |
|                    |         |         |             |             | eral users | cooperate |     | to find a | solution | for | a techni- |
datasetshowthattherecurrentneuralnetwork
|     |     |     |     |     | cal issue | contributed |     | by another | user. | Each | agent |
| --- | --- | --- | --- | --- | --------- | ----------- | --- | ---------- | ----- | ---- | ----- |
basedmodelsofourframeworksrobustlypre-
mighthaveonepartofthesolution,andthesepieces
dictaddresseesandresponsesinconversations
|     |     |     |     |     | have to | be combined |     | through | conversation |     | in order |
| --- | --- | --- | --- | --- | ------- | ----------- | --- | ------- | ------------ | --- | -------- |
withalargenumberofagents.
tocomeupwiththewholesolution.
|     |     |     |     |     | A unique | issue | of  | such multi-party |     | conversations |     |
| --- | --- | --- | --- | --- | -------- | ----- | --- | ---------------- | --- | ------------- | --- |
1 Introduction
|     |     |     |     |     | is addressing, |     | a behavior | whereby |     | interlocutors | in- |
| --- | --- | --- | --- | --- | -------------- | --- | ---------- | ------- | --- | ------------- | --- |
Shorttextconversation(STC)hasbeengainingpop- dicate to whom they are speaking (Jovanovic´ and
ularity: givenaninputmessage,predictanappropri- Akker, 2004; Akker and Traum, 2009). In face-
ate response in a single-round, two-party conversa- to-face communication, the basic clue for speci-
tion(Wangetal.,2013;Shangetal.,2015). Model- fying addressees is turning one’s face toward the
ing STC is simpler than modeling a complete con- addressee. In contrast, in voice-only or text-
versation, but instantly helps applications such as basedcommunication,theexplicitdeclarationofad-
dressee’snamesismorecommon.
chat-botsandautomaticshort-messagereplies(Jiet
| al.,2014). |     |     |     |     | In this | work, | we  | tackle addressee |     | and | response |
| ---------- | --- | --- | --- | --- | ------- | ----- | --- | ---------------- | --- | --- | -------- |
Beyond two-party conversations, there is also a selection for multi-party conversation: given a con-
need for modeling multi-party conversation, a form text, predict an addressee and response. As Fig-
of conversation with several interlocutors convers- ure 1 shows, a system is required to select an ad-
ing with each other (Traum, 2003; Dignum and dressee from the agents appearing in the previous
Vreeswijk,2003;UthusandAha,2013). Forexam- contextandaresponsefromafixedsetofcandidate
ple, in the Ubuntu Internet Relay Chat (IRC), sev- responses(Section3).
2133
Proceedingsofthe2016ConferenceonEmpiricalMethodsinNaturalLanguageProcessing,pages2133–2143,
Austin,Texas,November1-5,2016.(cid:13)c2016AssociationforComputationalLinguistics

The key challenge for predicting appropriate ad- proaches utilize statistical models on top of heuris-
dressees and responses is to jointly capture who tic rules or templates (Levin et al., 2000; Young et
is talking about what at each time step in a con- al.,2010;Walkeretal.,2003),theyapplystatistical
text. For jointly modeling the speaker-utterance in- machine translation based techniques without such
formation, we present two modeling frameworks: heuristics, which leads to recent work utilizing the
1) static modeling and 2) dynamic modeling (Sec- SMT-basedtechniqueswithneuralnetworks(Shang
tion 5). While speakers are represented as fixed et al., 2015; Vinyals and Le, 2015; Sordoni et al.,
vectors in the static modeling, they are represented 2015;Serbanetal.,2016).
as hidden state vectors that dynamically change As another popular approach, retrieval-based
with time steps in the dynamic modeling. In prac- techniques are used to retrieve candidate responses
tice, our models trained for the task can be applied fromarepositoryandreturnthehighestscoringone
to retrieval-based conversation systems, which re- withtherankingmodel(Jietal., 2014; Wangetal.,
trievescandidateresponsesfromalarge-scalerepos- 2015;Huetal.,2014;Wangetal.,2013;LuandLi,
itorywiththematchingmodelandreturnsthehigh- 2013). Stemmingfromthisapproach,thenextutter-
estscoringonewiththerankingmodel(Wangetal., anceclassification(NUC)taskhasbeenproposed,in
2013;Jietal.,2014;Wangetal.,2015). Ourtrained which a system is required to select an appropriate
models work as the ranking model and allow the responsefromafixedsetofcandidates(Loweetal.,
conversation system to produce addressees as well 2015;Kadlecetal.,2015). TheNUCisregardedas
asresponses. focusing on the ranking problem of retrieval-based
Toevaluatethetrainedmodels,weprovideacor- system, since it omits the candidate retrieving step.
pus and dataset. By exploiting Ubuntu IRC Logs1, ThemeritofNUCisthatitallowsustoeasilyevalu-
we build a large-scale multi-party conversation cor- atethemodelperformanceonthebasisofaccuracy.
pus, and create a dataset from it (Section 6). Our Our proposed addressee and response selection
experimentsonthedatasetshowthemodelsinstanti- task is an extension of the NUC. We generalize the
atedbythestaticanddynamicmodelingoutperform task by integrating the addressee detection, which
a strong baseline. In particular, the model based on has been regarded as a problematic issue in multi-
thedynamicmodelingrobustlypredictsappropriate party conversation (Traum, 2003; Jovanovic´ and
addressees and responses even if the number of in- Akker, 2004; Uthus and Aha, 2013). Basically,
terlocutorsinaconversationincreases.2 the addressee detection has been tackled in the
Wemakethreecontributionsinthiswork: spoken/multimodal dialog system research, and the
modelslargelyrelyonacousticsignalorgazeinfor-
1. We formalize the task of addressee and re- mation (Jovanovic´ et al., 2006; Akker and Traum,
sponseselectionformulti-partyconversation. 2009;RavuriandStolcke,2014). Thiscurrentwork
isdifferentfromsuchpreviousworkinthatourmod-
2. We present modeling frameworks and the per-
elspredictaddresseeswithonlytextualinformation.
formancebenchmarksforthetask.
For predicting addressees or responses, how the
context is encoded is crucial. In single-round con-
3. Webuildalarge-scalemulti-partyconversation
versation, a system is expected to encode only one
corpusanddatasetforthetask.
utterance as a context (Ritter et al., 2011; Wang et
al., 2013). In contrast, in multi-turn conversation,
2 RelatedWork
a system is expected to encode multiple utterances
This work follows in the footsteps of Ritter et al. (Shang et al., 2015; Lowe et al., 2015). Very re-
(2011), who tackled the response generation prob- cently, individual personalities have been encoded
lem: given a context, generate an appropriate re- asdistributedembeddingsusedforresponsegenera-
sponse. While previous response generation ap- tionintwo-partyconversation(Lietal.,2016). Our
workisdifferentfromthatworkinthatourproposed
1http://irclogs.ubuntu.com/
2Our code, corpus, and dataset are publicly available at personality-independent representation allows us to
https://github.com/hiroki13/response-ranking handlenewagentsunseeninthetrainingdata.
2134

Type Notation To predict an addressee a as a target output, we
selectanagentfromasetoftheagentsappearingin
RespondingAgent a res a context ( ). Note that a ground-truth addressee
Input Context A C
C isalwaysincludedin ( ). Topredictanappropri-
CandidateResponses A C
R ate response r, we select a response from a set of
Addressee a ( )
Output ∈A C candidate responses , which consists of Q candi-
Response r R
∈R dates:
= r , ,r
Table1:NotationsfortheARStask. 1 Q
R { ··· }
r = (w , ,w )
3 AddresseeandResponseSelection
q q,1
···
q,Nq
where r is a candidate response, which consists of
Weproposeandformalizethetaskofaddresseeand q
N tokens, andw isantokenindexinthevocab-
response selection (ARS) for multi-party conversa- q q,n
ulary .
tion. The ARS task assumes the situation where a V
responding agent gives a response to an addressee
4 DualEncoderModels
followingacontext.3
Our proposed models are extensions of the dual
Notation encoder (DE) model in (Lowe et al., 2015). The
Table 1 shows the notations for the formalization. DEmodelconsistsoftworecurrentneuralnetworks
Wedenotevectorswithboldlower-case(e.g. x t ,h), (RNN) that respectively compute the vector repre-
matriceswithboldupper-case(e.g. W,H a ),scalars sentationofaninputcontextandcandidateresponse.
with italic lower-case or upper-case (e.g. a m , Q), A generic RNN, with input x t R dw and recur-
∈
andsetswithbolditaliclower-caseorcursiveupper- rentstateh t R dh,isdefinedas:
∈
case(e.g. x, )letters.
C
h = f(h ,x ) = π(W h +W x ) (1)
t t 1 t h t 1 x t
− −
Formalization
Given an input conversational situation x, an ad- whereπisanon-linearfunction,W x R dh× dw isa
∈
dresseeaandaresponser arepredicted: parametermatrixforx t ,W h R dh× dh isaparam-
∈
eter matrix for h , and the recurrence is seeded
t 1
GIVEN : x = (a
res
, , )
with the 0 vector,
−
i.e. h = 0. The recurrent state
C R 0
h actsasacompactsummaryoftheinputsseenup
t
PREDICT : a,r
totimestept.
wherea res isarespondingagent, isacontextand In the DE model, each word vector of the con-
C
is a set of candidate responses. The context is text and the response r is consumed by each
q
R C C
a sequence of previous utterances up to the current RNN, and is then summarized into the context vec-
timestepT: torh c R dh andtheresponsevectorh q R dh. Us-
∈ ∈
ingthesevectors,themodelcalculatestheprobabil-
= (u , ,u )
C
a1,1
···
aT,T ity that the given candidate response is the ground-
truthresponsegiventhecontextasfollows:
whereu isanutterancegivenbyanagenta ata
at,t t
timestept. Eachutteranceu isasequenceofN
at,t t Pr(y(r ) = 1 ,r ) = σ(hTWh ) (2)
tokens: q |C q c q
where y is a binary function mapping from r to
u = (w , ,w ) q
at,t at,t,1
···
at,t,Nt
0,1 , in which 1 represents the ground-truth sam-
{ }
wherew isatokenindexinthevocabulary . ple and 0 represents the false one, σ is the logistic
3Inactu
a
a
t
l
,t
s
,n
ituations,responsescanbeaddressedtomul
V
tiple
sigmoid function, and W
∈
R dh× dh is a parameter
matrix. Asextensionsofthismodel,weproposeour
agents.Inthiswork,weassumethesituationwhereonespecific
agentcanbetheaddresseeofaresponse. multi-partyencodermodels.
2135

5 Multi-PartyEncoderModels
For capturing multi-party conversational streams,
wejointlyencodewhoisspeakingwhatateachtime
step. Eachagentanditsutteranceareintegratedinto
thehiddenstatesofanRNN.
We present two multi-party modeling frame-
works: (i) static modeling and (ii) dynamic mod-
eling, both of which jointly utilize agent and ut-
terance representation for encoding multiple-party
conversation. What distinguishes the models is that
whiletheagentrepresentationinthestaticmodeling
frameworkisfixed,theoneinthedynamicmodeling Figure2:Illustrativeexampleofourstaticmodel.
frameworkchangesalongwitheachtimesteptina
conversation.
andresponseareselectedasfollows:
ModelingFrameworks aˆ = argmaxPr(y(a ) = 1 x) (5)
p
|
Asan instance of thestatic modeling, we propose a
ap
∈A
(
C
)
staticmodeltocapturethespeaking-ordersofagents rˆ = argmaxPr(y(r q ) = 1 x) (6)
r |
inconversation. Asaninstanceofthedynamicmod- q ∈R
eling, we propose a dynamic model using an RNN whereaˆ isthehighestprobabilityaddresseeofaset
to track agent states. Note that the agent represen- of agents in the context ( ), and rˆ is the highest
A C
tations are independent of each personality (unique probability response of a set of candidate responses
user). The personality-independent representation .
R
allowsustohandlenewagentsunseeninthetraining
5.1 AStaticModel
data.
Formally, similar to Eq. 2, both of the models Inthestaticmodel,agentmatrixAisdefinedforthe
calculatetheprobabilitythattheaddresseea orre- agentvectorsinEqs. 3and4. Thisagentmatrixcan
p
sponser istheground-truthgiventheinputx: bedefinedarbitrarily. WedefinetheagentmatrixA
q
on the basis of agents’ speaking orders. Intuitively,
Pr(y(a ) = 1 x) = σ([a ; h ]TW a ) (3) the agents that spoke in recent time steps are more
p res c a p
|
likelytobeanaddressee. Ourstaticmodelcaptures
Pr(y(r ) = 1 x) = σ([a ; h ]TW h ) (4) suchproperty.
q res c r q
|
The static model is shown in Figure 2. First,
where y is a binary function mapping from a p or agents in the context ( ) and a responding agent
A C
r q to { 0,1 } , in which 1 represents the ground-truth a res are sorted in descending order based on each
sample and 0 represents the false one. The func- latest speaking time. Then the order is assigned as
i t s io a n r σ es i p s o t n h d e in lo g gi a s g ti e c nt si v g e m ct o o i r d , f a u p nctio R n d . a a i r s es a ∈ can R d d i a - a In n t a h g e e t n a t b i l n e d s e h x o a w m ni ∈ nF (1 ig , u ·· re · , 2 | , A th ( e C r ) e | ) sp to on e d a i c n h g a a g g e e n n t t .
∈
dateaddresseevector,h c R dh isacontextvector, (represented as SYSTEM) has the agent index 1 be-
∈
h q R dh isacandidateresponsevector. Thesevec- cause he spoke at the most recent time step t = 6.
∈
tors are respectively defined in each model. W a Similarly,User 1hastheindex2becausehespoke
∈
R (da+dh) × dh isaparametermatrixfortheaddressee atthesecondmostrecenttimestept = 5,andUser
selection probability, and W r R (da+dh) × dh is a 2hastheindex3becausehespokeatthethirdt = 3.
∈
parameter matrix for the response selection proba- Each speaking-order index a is associated with
m
bility. These model parameters are learned during thea -thcolumnoftheagentmatrixA:
m
training.
OnthebasisofEqs. 3and4,aresultingaddressee a = A[ ,a ]
m m
∗
2136

ance vector at each time step. Note that the states
of the agents that are not speaking at the time are
updatedbyzerovectors.
Formally, each column of A corresponds to an
t
agentstatevector:
a = A [ ,a ]
m,t t m
∗
whereanagentstatevectora ofanagenta ata
m,t m
time step t is the a -th column of the agent matrix
m
A .
t
Eachvectorofthematrixisupdatedat eachtime
Figure3:Illustrativeexampleofourdynamicmodel.
step, as shown in Figure 3. An agent state vector
a m,t R da foreachagenta m ateachtimesteptis
∈
recurrentlycomputed:
Similarly,arespondingagentvectora andacan-
res
didate addressee vector a in Eqs. 3 and 4 are re-
p
a = g(a ,u ), a = 0
spectively extracted from A, i.e. a = A[ ,a ] m,t m,t 1 m,t m,0
res ∗ res −
anda = A[ ,a ].
p p
Consuming ∗ theagentvectors,anRNNupdatesits where u m,t ∈ R dw is a summary vector of an ut-
teranceofanagenta andcomputedwithanRNN.
hiddenstate. Forexample, atthetimestep t = 1in m
As the transition function g, we use the GRU. For
Figure2,theagentvectora ofUser1isextracted
1
example, at a time step t = 2 in Figure 3, the agent
from A on the basis of agent index 2 and then con-
statevectora isinfluencedbyitsutterancevector
sumedbytheRNN.Then, theRNNconsumeseach 1,2
u andupdatedfromthepreviousstatea .
word vector w of User 1’s utterance. By consum- 1,2 1,1
TheagentmatrixupdateduptothetimestepT is
ingtheagentvectorbeforewordvectors, themodel
denoted as A , which is max-pooled and used as a
can capture which agent speaks the utterance. The T
summarizedcontextvector:
laststateoftheRNNisregardedash . Asthetran-
c
sitionfunctionf ofRNN(Eq. 1),weusetheGated
h = max A [i]
Recurrent Unit (GRU) (Cho et al., 2014; Chung et c T
i
al.,2014).
For the candidate response vector h q , each word The agent matrix A T is also used for a responding
vector(w q,1 ,
···
,w q,Nq )intheresponser q issum- agent vector a res and a candidate addressee vector
marizedwiththeRNN.Usingthesevectorsa res ,a p , a p ,i.e. a res = A T [ ∗ ,a res ]anda p = A T [ ∗ ,a p ]. r q
h c ,andh q ,wepredictanextaddresseeandresponse issummarizedintoaresponsevectorh q inthesame
withtheEqs. 3and4. wayasthestaticmodel.
5.3 Learning
5.2 ADynamicModel
We train the model parameters by minimizing the
In the static model, agent representation A is a
jointlossfunction:
fixedmatrixthatdoesnotchangeinaconversational
stream. In contrast, in the dynamic model, agent
λ
representation A t tracks each agent’s hidden state L (θ) = α L a (θ)+(1 − α) L r (θ)+ 2|| θ || 2
whichdynamicallychangeswithtimestepst.
Figure 3 shows the overview of the dynamic where isthelossfunctionfortheaddresseeselec-
a
L
model. Initially,wesetazeromatrixasinitialagent tion, is the loss function for the response selec-
r
L
stateA ,andeachcolumnvectoroftheagentmatrix tion, α is the hyper-parameter for the interpolation,
0
corresponds to an agent hidden state vector. Then, and λ is the hyper-parameter for the L2 weight de-
each agent state is updated by consuming the utter- cay.
2137

|     |     |     |     |     |     |     |               |                                  | Corpus |      |        | Dataset       |         |
| --- | --- | --- | --- | --- | --- | --- | ------------- | -------------------------------- | ------ | ---- | ------ | ------------- | ------- |
|     |     |     |     |     |     |     |               |                                  |        |      | Train  | Dev           | Test    |
|     |     |     |     |     |     |     | No. ofDocs    |                                  |        | 7355 | 6,606  |               | 367 382 |
|     |     |     |     |     |     |     | No. ofUtters  |                                  | 2.4M   |      | 2.1M   | 13.2k         | 15.1k   |
|     |     |     |     |     |     |     | No. ofWords   |                                  | 27.0M  |      | 23.8M  | 1.5M          | 1.7M    |
|     |     |     |     |     |     |     | No. ofSamples |                                  |        | -    | 665.6k | 45.1k         | 51.9k   |
|     |     |     |     |     |     |     | Avg. W./U.    |                                  |        | 11.1 | 11.1   | 11.2          | 11.3    |
|     |     |     |     |     |     |     | Avg. A./D.    |                                  |        | 26.8 | 26.3   | 30.68         | 32.1    |
|     |     |     |     |     |     |     | Table2:       | Statisticsofthecorpusanddataset. |        |      |        | “Docs”isdocu- |         |
ments,“Utters”isutterances,“W./U.”isthenumberofwords
perutterance,“A./D.”isthenumberofagentsperdocument.
|     |     |     |     |     |     |     | Then, from | the | logs, | we  | extract | and add | addressee |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ----- | --- | ------- | ------- | --------- |
Figure4:Theflowofthecorpusanddatasetcreation.Fromthe
|     |     |     |     |     |     |     | information | to  | the corpus. |     | In the | final | step, we set |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ----------- | --- | ------ | ----- | ------------ |
originallogs,weextractaddresseeIDsandaddthemtothecor-
|     |     |     |     |     |     |     | candidateresponsesandlabelsasthedataset. |     |     |     |     |     | Table |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | --- | ----- |
pus.Asthedataset,weaddcandidateresponsesandthelabels.
2showsthestatisticsofthecorpusanddataset.
For addressee and response selection, we use the 6.1 UbuntuIRCLogs
cross-entropylossfunctions:
|     |     |     |      |          |     |     | The Ubuntu     | IRC      | Logs | is     | a collection | of        | logs from |
| --- | --- | --- | ---- | -------- | --- | --- | -------------- | -------- | ---- | ------ | ------------ | --------- | --------- |
|     |     |     |      |          |     |     | Ubuntu-related |          | chat | rooms. | In           | each chat | room, a   |
|     | (θ) | =   | [log | Pr(y(a+) | = 1 | x)  |                |          |      |        |              |           |           |
|     | a   |     |      |          |     |     | number         | of users | chat | on and | discuss      | various   | topics,   |
|     | L   | −   |      |          |     | |   |                |          |      |        |              |           |           |
n
|     |       |        | ∑    |          |           |     | mainly                                    | related | to technical |             | support | with | Ubuntu is- |
| --- | ----- | ------ | ---- | -------- | --------- | --- | ----------------------------------------- | ------- | ------------ | ----------- | ------- | ---- | ---------- |
|     |       | +log(1 |      | Pr(y(a − | ) = 1 x)] |     | sues.                                     |         |              |             |         |      |            |
|     |       |        | −    |          | |         |     |                                           |         |              |             |         |      |            |
|     |       |        |      | Pr(y(r+) |           |     | Thelogsareputtogetherintoonefileperdayfor |         |              |             |         |      |            |
|     | r (θ) | =      | [log |          | = 1       | x)  |                                           |         |              |             |         |      |            |
|     | L     | −      |      |          |           | |   |                                           |         |              |             |         |      |            |
|     |       |        |      |          |           |     | each room.                                | Each    | file         | corresponds |         | to   | a document |
n
∑
+log(1 Pr(y(r ) = 1 x)] . In a document, one line corresponds to one log
|     |     |     |     | −   |     |     | D        |         |      |     |          |     |             |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------- | ---- | --- | -------- | --- | ----------- |
|     |     |     | −   |     | |   |     | given by | a user. | Each | log | consists | of  | three items |
where x is the input set for the task, i.e. x = (Time,UserID,Utterance). Using such informa-
),a+
(a res , , isaground-truthaddressee,a − isa tion,wecreateamulti-partyconversationcorpus.
C R
| false | addressee, |     | r+ is | a ground-truth | response, | and |     |     |     |     |     |     |     |
| ----- | ---------- | --- | ----- | -------------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
r is a false response. As a false addressee a , 6.2 TheMulti-PartyConversationCorpus
|     | −   |     |     |     |     | −   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
we pick up and use the addressee with the high- To pick up only the documents written in En-
est probability from the set of candidate addressees glish,weusealanguagedetectionlibrary(Nakatani,
except the ground-truth one ( ( ) a+). As a 2010). Then, we removethesystem logs from each
|     |     |     |     |     | A C \ |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
false response, we randomly pick up and use a re- documentandleaveonlyuserlogs. Forsegmenting
sponse from the set of candidate responses except thewordsineachutterance,weuseawordtokenizer
| theground-truthone( |     |     |     | r+). |     |     | (TreebankWordTokenizer) |     |     |     |     |     |             |
| ------------------- | --- | --- | --- | ---- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | ----------- |
|                     |     |     |     |      |     |     |                         |     |     |     |     | of  | the Natural |
R\
|     |     |     |     |     |     |     | Language | Toolkit4. |     | Using | the preprocessed |     | docu- |
| --- | --- | --- | --- | --- | --- | --- | -------- | --------- | --- | ----- | ---------------- | --- | ----- |
6 CorpusandDataset
|     |     |     |     |     |     |     | ments, | we create | a   | corpus, | whose | row | consists of |
| --- | --- | --- | --- | --- | --- | --- | ------ | --------- | --- | ------- | ----- | --- | ----------- |
thethreeitems(UserID,Addressee,Utterance).
| Our            | goal | is to | provide | a multi-party | conversation |            |        |         |        |       |      |          |          |
| -------------- | ---- | ----- | ------- | ------------- | ------------ | ---------- | ------ | ------- | ------ | ----- | ---- | -------- | -------- |
|                |      |       |         |               |              |            | First, | the IDs | of the | users | in a | document | are col- |
| corpus/dataset |      | that  | can     | be used       | over a       | wide range |        |         |        |       |      |          |          |
lectedintotheuserIDlistbyreferringtotheUserID
ofconversationresearch,suchasturn-takingmodel-
|     |     |     |     |     |     |     | in each | log. Then, | as  | the | addressee | user | ID, we ex- |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | --- | --- | --------- | ---- | ---------- |
ing(RauxandEskenazi,2009)anddisentanglement
modeling(ElsnerandCharniak,2010),aswellasfor tractthefirstwordofeachutterance. IntheUbuntu
|                                            |     |         |          |          |          |               | IRC Logs,             | users | follow | the    | name | mention | conven-      |
| ------------------------------------------ | --- | ------- | -------- | -------- | -------- | ------------- | --------------------- | ----- | ------ | ------ | ---- | ------- | ------------ |
| the                                        | ARS | task.   | Figure   | 4 shows  | the flow | of the cor-   |                       |       |        |        |      |         |              |
|                                            |     |         |          |          |          |               | tion (Uthus           | and   | Aha,   | 2013), | in   | which   | they express |
| pus                                        | and | dataset | creation | process. | We       | firstly crawl |                       |       |        |        |      |         |              |
| UbuntuIRCLogsandpreprocesstheobtainedlogs. |     |         |          |          |          |               | 4http://www.nltk.org/ |       |        |        |      |         |              |
2138

their addressee by mentioning the addressee’s user selected. Intheaddressee/responseselection,were-
ID at the beginning of the utterance. By exploiting gardtheanswerascorrectiftheaddressee/response
| the name                                     | mentions, | if        | the first | word of  | each     | utter- | iscorrectlyselected. |     |     |     |     |     |     |
| -------------------------------------------- | --------- | --------- | --------- | -------- | -------- | ------ | -------------------- | --- | --- | --- | --- | --- | --- |
| ance is identical                            |           | to a user | ID in     | the user | ID list, | we     |                      |     |     |     |     |     |     |
| extracttheaddresseeIDandthencreateatablecon- |           |           |           |          |          |        | Optimization         |     |     |     |     |     |     |
sisting of (UsetID,Addressee,Utterance). In Themodelsaretrainedbybackpropagationthrough
| the case | that addressee |     | IDs are | not explicitly |     | men- |               |     |       |        |     |              |     |
| -------- | -------------- | --- | ------- | -------------- | --- | ---- | ------------- | --- | ----- | ------ | --- | ------------ | --- |
|          |                |     |         |                |     |      | time (Werbos, |     | 1990; | Graves | and | Schmidhuber, |     |
tioned at the beginning of the utterance, we do not 2005). For the backpropagation, we use stochastic
extractanything. gradient descent (SGD) with a mini-batch training
|     |     |     |     |     |     |     | method. | The | mini-batch | size | is  | set to 128. | The |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | ---------- | ---- | --- | ----------- | --- |
6.3 TheARSDataset hyper-parameterαfortheinterpolationbetweenthe
By exploiting the corpus, we create a dataset for twolossfunctions(Section5.3)issetto0.5. Forthe
the ARS task. If the line of the corpus includes L2 weight decay, the hyper-parameter λ is selected
an addressee ID, we regard it as a sample for the from 0.001,0.0005,0.0001 .
|     |     |     |     |     |     |     | {   |     |     |     | }   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
task. Asthegroundtruthaddresseesandresponses,
|     |     |     |     |     |     |     | Parameters |     | of the | models | are | randomly | ini- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------ | ------ | --- | -------- | ---- |
westraightforwardlyusetheobtainedaddresseeIDs tialized over a uniform distribution with support
andthepreprocessedutterances. [ 0.01,0.01]. To update parameters, we use Adam
−
As false responses, we sample utterances else- (KingmaandBa,2014)withthedefaultsettingsug-
where within a document. This document-within gested by the authors. As the word embeddings,
sampling method makes the response selection task we used the 300 dimension vectors pre-trained by
method5.
more difficult than the random sampling GloVe6 (Pennington et al., 2014). To avoid over-
| One reason | for | this is that | common | or  | similar | top- |              |      |         |     |       |            |        |
| ---------- | --- | ------------ | ------ | --- | ------- | ---- | ------------ | ---- | ------- | --- | ----- | ---------- | ------ |
|            |     |              |        |     |         |      | fitting, the | word | vectors | are | fixed | across all | exper- |
ics in a document are often discussed and the used iments. The hidden dimensions of parameters are
words tend to be similar, which makes the word- set to d = 300 and d = 50 in the both models,
|     |     |     |     |     |     |     |     | w   |     | h   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
based features for the task less effective. We par- andd a issetto300inthestaticmodeland50inthe
| titioned | the dataset | randomly |     | into a | training | set | dynamicmodel. |     |     |     |     |     |     |
| -------- | ----------- | -------- | --- | ------ | -------- | --- | ------------- | --- | --- | --- | --- | --- | --- |
(90%),adevelopmentset(5%)andatestset(5%). Toidentifythebesttrainingepochandmodelcon-
figuration,weusetheearlystoppingmethod(Yaoet
7 Experiments
|     |     |     |     |     |     |     | al., 2007). | In  | this method, |     | if the | best accuracy | of  |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------------ | --- | ------ | ------------- | --- |
ADR-RES
We provide performance benchmarks of our learn- on the development set has not been up-
|                   |     |        |           |              |     |     | dated for | consecutive     |     | 5 epochs, | training | is         | stopped |
| ----------------- | --- | ------ | --------- | ------------ | --- | --- | --------- | --------------- | --- | --------- | -------- | ---------- | ------- |
| ing architectures |     | on the | addressee | and response |     | se- |           |                 |     |           |          |            |         |
|                   |     |        |           |              |     |     | and the   | best performing |     | model     | is       | picked up. | The     |
lection(ARS)taskformulti-partyconversation.
|     |     |     |     |     |     |     | maxepochsissetto30, |     |     | whichissufficientforcon- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | ------------------------ | --- | --- | --- |
vergence.
7.1 ExperimentalSetup
Datasets
ImplementationDetails
| Weusethecreateddatasetfortheexperiments. |     |     |     |     |     | The |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
numberofcandidateresponsesRES-CAND( )is For computational efficiency, we limit the length of
|             |     |     |     |     | |R| |     |           |          |          | =       | (u  | ,      | ,u ),  |
| ----------- | --- | --- | --- | --- | --- | --- | --------- | -------- | -------- | ------- | --- | ------ | ------ |
| setto2or10. |     |     |     |     |     |     | a context | as       | T Nc+1:T |         | T   | Nc+1   | T      |
|             |     |     |     |     |     |     |           | C        | C −      |         |     | − ···  |        |
|             |     |     |     |     |     |     | where N   | , called | context  | window, |     | is the | number |
c
|                   |     |     |     |     |     |     | of utterances | prior | to  | a time | step | t. We set | N to |
| ----------------- | --- | --- | --- | --- | --- | --- | ------------- | ----- | --- | ------ | ---- | --------- | ---- |
| EvaluationMetrics |     |     |     |     |     |     |               |       |     |        |      |           | c    |
We evaluate performance by accuracies on 5,10,15 . In addition, we truncate the utterances
|                |     |                    |     |      |           |     | {                                | }   |     |     |     |          |     |
| -------------- | --- | ------------------ | --- | ---- | --------- | --- | -------------------------------- | --- | --- | --- | --- | -------- | --- |
|                |     |                    |     |      |           |     | andresponsesatamaximumof20words. |     |     |     |     | Forbatch |     |
| three aspects: |     | addressee-response |     | pair | selection |     |                                  |     |     |     |     |          |     |
processing,wezero-padthemsothatthenumberof
| (ADR-RES),      | addressee |        | selection | (ADR),             | and     | re- |             |           |                   |       |        |                 |         |
| --------------- | --------- | ------ | --------- | ------------------ | ------- | --- | ----------- | --------- | ----------------- | ----- | ------ | --------------- | ------- |
|                 |           | (RES). |           |                    |         |     | words is    | constant. | Out-of-vocabulary |       |        | words           | are re- |
| sponse          | selection |        | In the    | addressee-response |         |     |             |           |                   |       |        |                 |         |
|                 |           |        |           |                    |         |     | placed with | <unk>,    |                   | whose | vector | is the averaged |         |
| pair selection, | we        | regard | the       | answer as          | correct | if  |             |           |                   |       |        |                 |         |
vectoroverallwordvectors.
| both the | addressee | and | the response | are | correctly |     |     |     |     |     |     |     |     |
| -------- | --------- | --- | ------------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
5Loweetal.(2015)adoptedtherandomsamplingmethod. 6http://nlp.stanford.edu/projects/glove/
2139

|     |     |     |     |         | RES-CAND=2 |     |     | RES-CAND=10 |     |     |
| --- | --- | --- | --- | ------- | ---------- | --- | --- | ----------- | --- | --- |
|     |     |     | N   | ADR-RES |            | ADR | RES | ADR-RES     | ADR | RES |
c
|     |     | Chance   | -   |     | 0.62  | 1.24  | 50.00 | 0.12  | 1.24  | 10.00 |
| --- | --- | -------- | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- |
|     |     |          | 5   |     | 36.97 | 55.73 | 65.68 | 16.34 | 55.73 | 28.19 |
|     |     | Baseline | 10  |     | 37.42 | 55.63 | 67.79 | 16.11 | 55.63 | 29.48 |
|     |     |          | 15  |     | 37.13 | 55.62 | 67.89 | 15.44 | 55.62 | 29.19 |
|     |     |          | 5   |     | 46.99 | 60.39 | 75.07 | 21.98 | 60.26 | 33.27 |
|     |     | Static   | 10  |     | 48.67 | 60.97 | 77.75 | 23.31 | 60.66 | 35.91 |
|     |     |          | 15  |     | 49.27 | 61.95 | 78.14 | 23.49 | 60.98 | 36.58 |
|     |     |          | 5   |     | 49.80 | 63.19 | 76.07 | 23.72 | 63.28 | 33.62 |
|     |     | Dynamic  | 10  |     | 53.85 | 66.94 | 78.16 | 25.95 | 66.70 | 36.14 |
|     |     |          | 15  |     | 54.88 | 68.54 | 78.64 | 27.19 | 68.41 | 36.93 |
Table 3: Benchmark results: accuracies on addressee-response selection (ADR-RES), addressee selection (ADR), and response
| selection(RES).N |     | cisthecontextwindow.Boldedarethebestpercolumn. |     |     |     |     |     |     |     |     |
| ---------------- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
BaselineModel
| We set                                | a baseline | using    | the term | frequency-inverse |           |     |     |     |     |     |
| ------------------------------------- | ---------- | -------- | -------- | ----------------- | --------- | --- | --- | --- | --- | --- |
| document                              | frequency  | (TF-IDF) |          | retrieval         | model     | for |     |     |     |     |
| theresponseselection(Loweetal.,2015). |            |          |          |                   | Wefirstly |     |     |     |     |     |
computetwoTF-IDFvectors,oneforacontextwin-
| dow and       | one                       | for a candidate |           | response.     | Then,    | we        |     |     |     |     |
| ------------- | ------------------------- | --------------- | --------- | ------------- | -------- | --------- | --- | --- | --- | --- |
| compute       | a cosine                  | similarity      | for       | these         | vectors, | and       |     |     |     |     |
| select the    | highest                   | scoring         | candidate |               | response | as        | a   |     |     |     |
| result.       | Fortheaddresseeselection, |                 |           | weadoptarule- |          |           |     |     |     |     |
| based method: |                           | to determine    | the       | agent         | that     | gives an  |     |     |     |     |
| utterance     | most                      | recently        | except    | a responding  |          | agent,    |     |     |     |     |
| which         | captures                  | the tendency    | that      | agents        |          | often re- |     |     |     |     |
Figure5:Accuraciesinaddressee-responseselectionusingdif-
spondtotheotherthatspokeimmediatelybefore.
ferentamountofsamplesfortraining.
7.2 Results
OverallPerformance
thedynamicmodelachievesaround0.5pointhigher
| Table3showstheempiricalbenchmarkresults. |       |          |          |         |     | The     | inaccuracy. |     |     |     |
| ---------------------------------------- | ----- | -------- | -------- | ------- | --- | ------- | ----------- | --- | --- | --- |
| dynamic                                  | model | achieves | the best | results | in  | all the |             |     |     |     |
metrics. The static model outperforms the baseline, EffectsoftheContextWindow
butisinferiortothedynamicmodel. In response selection, a performance boost of our
In addressee selection (ADR), the baseline model proposed models is observed for the context win-
achievesaround55%inaccuracy. Thismeansthatif dow = 10 over = 5. Comparing the results
|     |     |     |     |     |     |     |     | c   | c   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     |     | N   | N   |     |
youselecttheagentsthatspokemostrecentlyasan ofthemodelswiththecontextwindow = 10and
c
N
addressee, the half of them are correct. Compared = 15,theperformanceisimprovedbutrelatively
|     |     |     |     |     |     |     | N c |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
withthebaseline,ourproposedmodelsachievebet- small, which suggests that the performance almost
ter results, which suggests that the models can se- reachestheconvergence. Inaddresseeselection,the
lect the correct addressees that spoke at more pre- performanceimprovementsofthestaticmodelwith
vious time steps. In particular, the dynamic model the broader context window is limited. In contrast,
achieves 68% in accuracy, which is 7 point higher in the dynamic model, a steady performance boost
thantheaccuracyofstaticmodel. isobserved,yieldinganincreaseofover5pointsbe-
In response selection (RES), our models outper- tweenN = 15andN = 5,
|          |           |          |      |     |        |        |     | c   | c   |     |
| -------- | --------- | -------- | ---- | --- | ------ | ------ | --- | --- | --- | --- |
| form the | baseline. | Compared | with | the | static | model, |     |     |     |     |
2140

No. ofAgents 2-5 6-10 11-15 16-20 21-30 31-100 101-305
No. ofSamples 3731 5962 5475 4495 5619 7956 18659
ADR-RES
Baseline 52.13 43.51 39.98 42.96 39.70 36.55 29.22
Static 64.17 55.92 50.72 53.04 48.69 49.61 42.86
Dynamic 66.90 57.73 54.32 55.64 51.61 55.88 52.14
ADR
Baseline 84.94 70.82 62.14 65.52 58.89 51.28 41.47
Static 86.33 74.37 66.12 68.54 63.43 59.24 50.99
Dynamic 87.64 76.48 69.99 72.21 66.90 66.78 62.11
RES
Baseline 60.71 61.24 64.51 65.58 67.93 71.66 71.38
Static 73.60 73.45 74.54 75.95 75.17 81.50 81.60
Dynamic 75.64 74.12 75.53 75.17 76.05 81.96 81.81
Table4:Performancecomparisonfordifferentnumbersofagentsappearinginthecontext.Thenumbersareaccuraciesonthetest
setwiththenumberofcandidateresponsesCAND-RES=2andthecontextwindowN
c
=15.
EffectsoftheSampleSize 8 Conclusion
Figure 5 shows the accuracy curves of addressee-
We proposed addressee and response selection for
response selection (ADR-RES) for different train-
multi-party conversation. Firstly, we provided the
ing sample sizes. We use 1/2, 1/4, and 1/8 of
formaldefinitionofthetask,andthencreatedacor-
the whole training samples for training. The results
pus and dataset. To present benchmark results, we
show that as the amount of the data increases, the
proposed two modeling frameworks, which jointly
performance of our models are improved and grad-
model speakers and their utterances in a context.
ually approaches the convergence. Remarkably, the
Experimental results showed that our models of the
performance of the dynamic models using the 1/8
frameworksoutperformabaseline.
samplesiscomparabletothatofthestaticmodelus-
Our future objective to tackle the task of predict-
ingthewholesamples.
ing whether to respond to a particular utterance. In
thiswork,weassumethatthesituationswherethere
EffectsoftheNumberofParticipants
is a specific addressee that needs an appropriate re-
Toshedlightontherelationshipbetweenthemodel
sponseandasystemisrequiredtorespond. Inactual
performanceandthenumberofagentsinmulti-party
multi-party conversation, however, a system some-
conversation, we investigate the effect of the num-
times has to wait and listen to the conversation that
ber of agents participating in each context. Table 4
other participants are engaging in without needless
compares the performance of the models for differ-
interruption. Hence, the prediction of whether to
entnumbersofagentsinacontext.
respond in a multi-party conversation would be an
In addressee selection, the performance of all
importantnextchallenge.
modelsgraduallygetsworseasthenumberofagents
in the context increases. However, compared with
Acknowledgments
thebaseline,ourproposedmodelssuppresstheper-
formance degradation. In particular, the dynamic We thank Graham Neubig, Yuya Taguchi, Ryosuke
modelpredictscorrectaddresseesmostrobustly. Kohita,AnderMartinez,themembersoftheNAIST
ComputationalLinguisticsLaboratory,themembers
In response selection, unexpectedly, the perfor-
ofIBMResearch-Tokyo,LongDuong, andthere-
mance of all the models gets better as the number
viewersfortheirhelpfulcomments.
ofagentsincreases. Detailedinvestigationonthein-
teraction between the number of agents and the re-
sponseselectioncomplexityisaninterestinglineof
futurework.
2141

References dataset for research in unstructured multi-turn dia-
logue systems. In Proceedings of SIGDIAL, pages
RieksAkkerandDavidTraum. 2009. Acomparisonof
285–294.
addressee detection methods for multiparty conversa-
Zhengdong Lu and Hang Li. 2013. A deep architec-
tions. InWorkshopontheSemanticsandPragmatics
tureformatchingshorttexts. InProceedingsofNIPS,
ofDialogue.
pages1367–1375.
KyunghyunCho,BartvanMerrienboer,CaglarGulcehre,
Shuyo Nakatani. 2010. Language detection library for
DzmitryBahdanau,FethiBougares,HolgerSchwenk,
java.
andYoshuaBengio. 2014. Learningphraserepresen-
tations using rnn encoder–decoder for statistical ma- Jeffrey Pennington, Richard Socher, and Christopher
chine translation. In Proceedings of EMNLP, pages Manning. 2014. Glove: Globalvectorsforwordrep-
1724–1734. resentation. In Proceedings of EMNLP, pages 1532–
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, 1543.
and Yoshua Bengio. 2014. Empirical evaluation of AntoineRauxandMaxineEskenazi. 2009. Afinite-state
gated recurrent neural networks on sequence model- turn-takingmodelforspokendialogsystems. InPro-
ing. arXivpreprintarXiv: 1412.3555. ceedingsofNAACL,pages629–637.
FrankPMDignumandGerardAWVreeswijk. 2003. To- SumanVRavuriandAndreasStolcke. 2014. Neuralnet-
wardsatestbedformulti-partydialogues. Advancesin work models for lexical addressee detection. In Pro-
AgentCommunication,pages212–230. ceedingsofINTERSPEECH,pages298–302.
Micha Elsner and Eugene Charniak. 2010. Disentan- AlanRitter, ColinCherry, andWilliamB.Dolan. 2011.
gling chat. Computational Linguistics, pages 389– Data-driven response generation in social media. In
409. ProceedingsofEMNL,pages583–593.
Alex Graves and Ju¨rgen Schmidhuber. 2005. Frame-
IulianVladSerban,AlessandroSordoni,YoshuaBengio,
wise phoneme classification with bidirectional lstm
Aaron Courville, and Joelle Pineau. 2016. Build-
and other neural network architectures. Neural Net-
ing end-to-end dialogue systems using generative hi-
works,18(5):602–610.
erarchical neural network models. In Proceedings of
BaotianHu,ZhengdongLu,HangLi,andQingcaiChen. AAAI,pages3776–3783.
2014. Convolutionalneuralnetworkarchitecturesfor
LifengShang,ZhengdongLu,andHangLi. 2015. Neu-
matchingnaturallanguagesentences. InProceedings
ralrespondingmachineforshort-textconversation. In
ofNIPS,pages2042–2050.
ProceedingsofACL/IJCNLP,pages1577–1586.
Zongcheng Ji, Zhengdong Lu, and Hang Li. 2014. An
AlessandroSordoni,MichelGalley,MichaelAuli,Chris
information retrieval approach to short text conversa-
Brockett, Yangfeng Ji, Margaret Mitchell, Jian-Yun
tion. arXivpreprintarXiv: 1408.6988.
Nie, Jianfeng Gao, and Bill Dolan. 2015. A
Natasa Jovanovic´ and op den Rieks Akker. 2004.
neural network approach to context-sensitive genera-
Towards automatic addressee identification in multi-
tion of conversational responses. In Proceedings of
partydialogues. InProceedingsofSIGDIAL.
NAACL/HLT,pages196–205.
Natasa Jovanovic´, op den Rieks Akker, and Anton Ni-
DavidTraum. 2003. Issuesinmultipartydialogues. Ad-
jholt. 2006. Addressee identification in face-to-face
vancesinAgentcommunication,pages201–211.
meetings. InProceedingsofEACL.
David C Uthus and David W Aha. 2013. Multipartic-
Rudolf Kadlec, Martin Schmid, and Jan Kleindiest.
ipant chat analysis: A survey. Artificial Intelligence,
2015. Improved deep learning baselines for ubuntu
pages106–121.
corpusdialogs. arXivpreprintarXiv: 1510.03753.
OriolVinyalsandV.QuocLe. 2015. Aneuralconversa-
Diederik P. Kingma and Jimmy Lei Ba. 2014. Adam:
tionalmodel. arXivpreprintarXiv: 1506.05869.
A method for stochastic optimization. arXiv preprint
arXiv: 1412.6980. Marilyn A Walker, Rashmi Prasad, and Amanda Stent.
Esther Levin, Roberto Pieraccini, and Wieland Eckert. 2003. A trainable generator for recommendations
2000. Astochasticmodelofhuman-machineinterac- in multimodal dialog. In Proceedings of INTER-
tionforlearningdialogstrategies. IEEETransactions SPEECH.Citeseer.
onSpeechandAudioProcessing,pages11–23. HaoWang, ZhengdongLu, HangLi, andEnhongChen.
Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, 2013. A dataset for research on short-text conversa-
andBillDolan. 2016. Apersona-basedneuralconver- tions. InProceedingsofEMNLP,pages935–945.
sationmodel. InProceedingsofACL. MingxuanWang,ZhengdongLu,HangLi,andQunLiu.
Ryan Lowe, Nissan Pow, Iulian V. Serban, and Joelle 2015. Syntax-based deep matching of short texts. In
Pineau. 2015. The ubuntu dialogue corpus: A large ProceedingsofIJCAI,pages1354–1361.
2142

| Paul J Werbos. | 1990.        | Backpropagation |                 | through | time:  |
| -------------- | ------------ | --------------- | --------------- | ------- | ------ |
| what it        | does and how | to do           | it. Proceedings |         | of the |
IEEE,78(10):1550–1560.
| Yuan Yao, | Lorenzo Rosasco, |     | and Andrea | Caponnetto. |     |
| --------- | ---------------- | --- | ---------- | ----------- | --- |
2007. Onearlystoppingingradientdescentlearning.
ConstructiveApproximation,26(2):289–315.
| Steve Young, | Milica | Gasˇic´, | Simon | Keizer, | Franc¸ois |
| ------------ | ------ | -------- | ----- | ------- | --------- |
Mairesse,JostSchatzmann,BlaiseThomson,andKai
| Yu. 2010. | The hidden | information |     | state | model: A |
| --------- | ---------- | ----------- | --- | ----- | -------- |
practicalframeworkforpomdp-basedspokendialogue
| management. | Computer | Speech | &   | Language, | pages |
| ----------- | -------- | ------ | --- | --------- | ----- |
150–174.
2143