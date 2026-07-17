| Multi-User |     | Chat | Assistant   |     | (MUCA):                   | a             | Framework |     | Using LLMs |     | to  |
| ---------- | --- | ---- | ----------- | --- | ------------------------- | ------------- | --------- | --- | ---------- | --- | --- |
|            |     |      | Facilitate  |     | Group                     | Conversations |           |     |            |     |     |
|            |     |      | ManqingMao∗ |     | PaishunTing∗,YijianXiang∗ |               |           |     |            |     |     |
MingyangXu∗,JuliaChen∗,JianzheLin∗
MicrosoftResearch
{manqing.mao,paishun.ting,yijianxiang,mingyangxu,juliachen,jianzhelin}@microsoft.com
Abstract
|     |     |     |     |     |     | Group Chat (Amy, Bob, Cindy, Dennis) |     |     | Design  | 3W Design  |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | ------- | ---------- | --- |
Recentadvancementsinlargelanguagemodels(LLMs)havepro- Challenges
|             |        |             |              |      |              | AAnnyyoonnee  kknnoowwss  hhooww  ttoo  eessttiimmaattee  tthhee  ccoosstt  ooff   |     |     |     | Dimensions |     |
| ----------- | ------ | ----------- | ------------ | ---- | ------------ | ---------------------------------------------------------------------------------- | --- | --- | --- | ---------- | --- |
| vided a new | avenue | for chatbot | development. | Most | existing re- |                                                                                    |     |     |     |            |     |
4202 tcO 4  ]LC.sc[  4v38840.1042:viXra AAAmmmyyy ttrraannssppoorrttaattiioonn  ffoorr  5500  ppeeooppllee??
search,however,hasprimarilycenteredonsingle-userchatbots Stuck
|     |     |     |     |     |     |     |     | NNoott  ssuurree....  @@bboobb?? | Conversation  |     | Timing  |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------- | ------------- | --- | ------- |
thatdetermine"What"toanswer.Thispaperhighlightsthecomplex- NNoo  iiddeeaa  ssoo  ffaarr...... CCCiiinnndddyyy Advancement
|     |     |     |     |     |     | BBBooobbb |     |     |     | Intelligence |     |
| --- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | ------------ | --- |
ityofmulti-userchatbots,introducingthe3W designdimensions: NNvvmm,,  lleett  uuss  ccoonnffiirrmm  tthhee  ppllaaccee.. Multi-threaded  (When)
|     |     |     |     |     |     |     |     |     | CCCiiinnndddyyy D i s cu s s io n   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- |
"What"tosay,"When"torespond,and"Who"toanswer.Addition- II  pprreeffeerr  MMoouunnttaaiinnXX  aass  iitt  iiss  cclloosseerr..
|     |     |     |     |     |     | AAAmmmyyy |     |     | M a n a g e m e n t |     | CCoonntteenntt   |
| --- | --- | --- | --- | --- | --- | --------- | --- | --- | ------------------- | --- | ---------------- |
ally,weproposedMulti-UserChatAssistant(MUCA),anLLM-based
|     |     |     |     |     |     |     | NNoo,,  MMoouunnttaaiinnYY  iiss  vvootteedd  llaasstt  ttiimmee.. |     | Responsiveness  | IInntteelllliiggeennccee |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------------------ | --- | --------------- | ------------------------ | --- |
frameworktailoredforgroupdiscussions.MUCAconsistsofthree CCCiiinnndddyyy ((WWhhaatt))
Requirement
mainmodules:Sub-topicGenerator,DialogAnalyzer,andConver- II  jjuusstt  sseeaarrcchheedd  oonnlliinnee,,  ttoo  eessttiimmaattee  tthhee  ccoosstt,,
|     |     |     |     |     |     | wwee  ccaann  ffiirrsstt  ddeecciiddee  wwhheerree  ttoo  ggoo  tthheenn   |                                        |                                                                    | Participation           |     |                 |
| --- | --- | --- | --- | --- | --- | -------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------ | ----------------------- | --- | --------------- |
|     |     |     |     |     |     | BBBooobbb c c oo                                                           | nn ss iidd ee rr   tt hh ee   tt yy pp | ee   oo ff   tt rr aa nn sspp oo rr tt aa ttii oo nn ,,   tt hh ee |    DDeennnniissEvenness | A   | d d r e s s e e |
sational Strategies Arbitrator. These modules jointly determine n n uu mm bb ee rr   oo ff   vv ee hh ii cc llee ss   nn ee ee dd ee dd   ff oo rr   55 00    pp ee oo pp ll ee..
I nt e ll i g e n c e
suitableresponsecontents,timings,andappropriateaddressees. Conflict  (Who)
|     |     |     |     |     |     | NNoo,,  nnooww  wwee  hhaavvee  ttoo  ddeecciiddee  tthhee  vveennuuee  bbyy  uuss   |     |     | Resolution |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------------------------------------------------------ | --- | --- | ---------- | --- | --- |
ThispaperfurtherproposesanLLM-basedMulti-UserSimulator AAAmmmyyy iinn  tthhiiss  cchhaatt  iinnsstteeaadd  ooff  tthhee  vvoottiinngg  llaasstt  ttiimmee..
(MUS)toeaseMUCA’soptimization,enablingfastersimulationof
Figure1:Adiagrammappingoutagroupchatsampletoits
conversationsbetweenthechatbotandsimulatedusers,andspeed-
associatedfivedesignchallengesandfurtherformulatedto
ingupMUCA’searlydevelopment.Ingoal-orientedconversations theproposed3W designdimensions.
withasmalltomediumnumberofparticipants,MUCAdemon-
strateseffectivenessintaskslikechiminginatappropriatetimings,
generatingrelevantcontent,andimprovinguserengagement,as WhileMUCAcanparticipateinconversationsofchit-chatnature,
shownbycasestudiesanduserstudies. wedemonstratedMUCA’seffectivenesswithbothcaseanduser
studies,focusingonseveralgoal-orientedtopics.Theevaluation
Keywords isusingquantitativemetricslikeuserengagement,conversation
LLM,Chatbot,Multi-user,Dialogue,UserStudy,CaseStudy evenness,andopinionconsensus.WealsomeasuredMUCA’sper-
formancebymetricslikeefficiency,conciseness,andusefulness
collectedfromuserfeedback,showingthatMUCAissuperiortoa
1 Introduction
baselinechatbot.Thehighlightsofourworkareasfollows:
Recentyearshaveseenasurgeofinterestinthefieldofchatbot
• WeshowthattheproposedMUCAenhancesthemulti-user
research.Largelanguagemodels(LLMs)likeGPTs[2,14,17]have
|     |     |     |     |     |     | chat | experience | by controlling | the 3W (What, | When, | Who) |
| --- | --- | --- | --- | --- | --- | ---- | ---------- | -------------- | ------------- | ----- | ---- |
emergedasapowerfultoolforchatbotdevelopment[8,23].How-
dimensionsthroughitsthreekeymodules(SubtopicGenerator,
ever,unlikesingle-userconversationchatbots,limitedresearchon
DialogAnalyzer,andUtteranceStrategiesArbitrator),enabling
groupconversationchatbotsrestrictstheirapplicationintaskslike
cohesiveconversationswithdeepercontextawareness.
brainstormingsessionsanddebates.
• WeproposeMUS,ausersimulatordesignedtomimicrealuser
ThispaperpresentsMulti-UserChatAssistant(MUCA),anLLM-
behaviorandsimulatedialoguesbetweenmultipleparticipants.
basedframeworkforgroupconversationchatbotswhich,asfar
MUSfacilitatestheoptimizationofMUCAbyenablingagentin-
| as the authors | are | aware of, | is the first LLM-based |     | framework |     |     |     |     |     |     |
| -------------- | --- | --------- | ---------------------- | --- | --------- | --- | --- | --- | --- | --- | --- |
teractionsthatincorporatethe"human-in-the-loop"approach.
| dedicatedtomulti-userconversations.Unlikesingle-userchatbots |     |     |     |     |     | •   |     |     |     |     |     |
| ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
WeevaluateMUCAthroughcasestudiesanduserstudiesacross
thatsimplydetermine"What"toanswerfollowingauser’sinputs,
varioustasksandgroupsizes.TheresultsshowthatMUCA
| multi-userchatbotshave3W |     | designdimensions,wheretheextra |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
consistentlyoutperformsabaselinechatbotintaskssuchas
twoare"When"toanswerand"Who"toanswer.Wedemonstrate
decision-making,problem-solving,andopendiscussions.
thatmanyofthechallengeslikeadvancingstuckconversationand
managingmulti-threadeddiscussioncanbemappedtothese3W 2 RelatedWork
dimensions.ToenablefastiterationanddevelopmentofMUCA,we
LLMs,suchasGPTs[2,14,16],havedemonstratedsuperiorperfor-
alsodeviseanLLM-basedMulti-UserSimulator(MUS)thatimproves
manceonvarioustasks.Moreover,thedevelopmentofLLMshas
overtimebyleveraginghuman-in-the-loopfeedback.
sparkedinterestinchatbotresearchandenabledvariousapplica-
∗Theseauthorscontributedequallytothisresearch. tionsbuiltaroundLLM-basedchatbots.
Single-userChatbots:Therehasbeensignificantexplorationof
thepre-trainingorfine-tuningofLLMsfortask-orienteddialogue

MUCA
MMUUSS  OOnnlliinnee//OOfffflliinnee  MMoodduullee MUS Sub-topics Generator
......
MMUUCCAA  OOnnlliinnee//OOfffflliinnee  MMoodduullee
- T1: determine ideal venue...
Set up a local book
|                        |     |     | MMuullttii--uusseerr  GGrroouupp  CChhaatt  SSeettttiinnggss |     |     |                 |     | - T2: collecting book donations... |     |     |
| ---------------------- | --- | --- | ------------------------------------------------------------ | --- | --- | --------------- | --- | ---------------------------------- | --- | --- |
| User Behavior Modeling |     |     |                                                              |     |     | exchange event. |     |                                    |     |     |
Sub-topics T
|     |     |     |     | Evaluation Mode |     | User-Input Info I |     |     |     |     |
| --- | --- | --- | --- | --------------- | --- | ----------------- | --- | --- | --- | --- |
Amy*: {Questioner, ...}
Bob*: {Topic Shifter, ...}
Dialog Analyzer
…
|                  |     |     |     |     |     | -   T 1 :   w e l l- d i | sc u s s e d  |                                    |     |     |
| ---------------- | --- | --- | --- | --- | --- | ------------------------ | ------------- | ---------------------------------- | --- | --- |
| Speaking Roles S |     | ... |     |     |     |                          |               | - T2: collecting book donations... |     |     |
|                  | r   |     |     |     |     | -   T 2 :   be i n g   d | is c u s s ed |                                    |     |     |
|                  |     |     |     |     |     | ...                      |               | Utterance Feature Extractor: T     |     | d   |
Amy*: {Laconic, ...}
Sub-topics Status
| Bob*: {Garrulous, ...} |     |     |     |     |     | Update: t |     |     |     |     |
| ---------------------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- |
- Topic: determine indoor course
| …   |     |                    |     |     |     |                         |                  | - -   A m y :   s u p p | o r t s   a r ts . . . |     |
| --- | --- | ------------------ | --- | --- | --- | ----------------------- | ---------------- | ----------------------- | ---------------------- | --- |
|     |     | U t te r a n c e   |     |     |     | S t a t is t i ca l   f | e a t u r e s :  |                         |                        |     |
Utterance Traits U L e n g t h  l - -   Bo b :   s u g g e s t s   y o g a . . .
|     |     | t u tt |     |     |     | {f r e q ,  l e n ,   N | e d ,  N i n g} | . . . |     |     |
| --- | --- | ------ | --- | --- | --- | ----------------------- | --------------- | ----- | --- | --- |
Participant Feature
|     |     |     |     |     |     | Extractor |     | Accumulative Summary Update: s |     |     |
| --- | --- | --- | --- | --- | --- | --------- | --- | ------------------------------ | --- | --- |
User Utterance
|     |            |     |     |     | TTiimmiinngg             | Conversational Strategy Arbitrator |     |     |     |     |
| --- | ---------- | --- | --- | --- | ------------------------ | ---------------------------------- | --- | --- | --- | --- |
|     | Generation |     |     |     | IInntteelllliiggeennccee |                                    |     |     |     |     |
((WWhheenn))
|     |     |     |     |     |     | Keep  | In-context  | Sub-topic  | Direct  |     |
| --- | --- | --- | --- | --- | --- | ----- | ----------- | ---------- | ------- | --- |
CCCooonnnttteeennnttt
|     |     |     |     |     | IIInnnttteeelllllliiigggeeennnccceee | Silent | Chime-in | Transition | Chatting |     |
| --- | --- | --- | --- | --- | ------------------------------------ | ------ | -------- | ---------- | -------- | --- |
(((WWWhhhaaattt)))
|                 |     |     | Simulated     | MUCA       |                      | Participation  |     | Conflict  | Initiative  |     |
| --------------- | --- | --- | ------------- | ---------- | -------------------- | -------------- | --- | --------- | ----------- | --- |
| Chat Snippets C | S   |     | Utterance utt | Response r | AA dd dd rr ee ss ss | ee ee          |     |           |             |     |
Development Mode II nntt ee llll ii gg ee nn cc ee Encouragement Resolution Summarization
((WWhhoo))
Figure2:Frameworkarchitecture,whichiscomposedoftheproposedMUCA(Sec.3.2)andMUS(Sec.3.3).TheMUCAis
periodicallyiteratedviatheinteractionwiththeproposedMUSinthedevelopmentmode,whilerealusersareinteractingwith
MUCAintheevaluationmode.Thetemporaryresultsinthegraydashboxesserveasexamples.
systems.Studiessuchas[3,9,23,29,32]haveemployedLLMs, 3 FrameworkArchitecture
pre-trainedorfine-tunedondialoguedata,todevelopdialogue 3.1 DesignDimensionsandChallenges
modelsorchatbotsforvariousdomainsandtasks,suchastravel
Inthissection,wedescribethe"3W"dimensionsformulti-userchat-
ticketsbookingorrestaurantreservation,etc.However,thesework
typicallyfocusonsingle-userscenarios. botsandthechallengesMUCAaddresses.Whilewebelieve"3W"
Multi-userChatbots:Mostresearchonmulti-partyormulti- dimensionsisappliedbroadlytovariedgroupchats,thechallenges
presentedcandifferbyscenario.Thispaperspecificallyfocuses
userdialoguesystems[6,7,15,21,34]havebeenfocusingontrain-
onchatbotsthatactasanassistantformulti-userconversations,
ingmodelsonmulti-partyconversationdatasetsforthefollowing
similartopriorrule-basedmulti-userchatbots[1,4,25].
tasks:addresseerecognition,speakeridentification,responseselec-
tionandgeneration.Beyondthesetasks,thereareotherimportant
dimensionsthathavebeenexploredwhendesigningmulti-party
|     |     |     |     |     | 3.1.1 | 3WDesignDimensions: |     | Single-userchatbotscenariosoften |     |     |
| --- | --- | --- | --- | --- | ----- | ------------------- | --- | -------------------------------- | --- | --- |
chatbots.Forexample,[5]proposedbalancedparticipationcommu-
followthe"adjacency-pair"structureinwhichoneutterancefrom
nicationstrategies,and[26]presentedfourmoderationstrategies
theuseranticipatesaresponsefromthechatbot[18].Therefore,
forplanningandnegotiatingjointappointments.[12]described
theprimarymetricforevaluatingsingle-userchatbotsfocuseson
fourfeaturesthatcanaidinfacilitatinggroupdiscussions.
content,orthe"What"dimension.Designingchatbotsformultiple
Differentfromtheabovework,ourMUCAhandlestheabove
usersisfarmorechallengingduetothe3W(What,When,Who)
tasksanddesigndimensionsinaunifiedframework.Theframework
dimensionsofthedesignspace,whicharethecontent,timing,and
isbasedonLLMs,suchasGPT-4,whichhasshowncomparable
recipientoftheresponse,asdetailedbelow:
performanceinzero-shotsettingstosupervisedmodelstrainedon
multi-partydatasets[24].Inaddition,MUCAreliesonprompting
| methods[28,30,31,33]toimprovethecapabilityofLLMsacross |     |     |     |     | •   |     |     |     |     |     |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ContentIntelligence("What"):Itrelatestowhatchatbots
variousdesigndimensions,avoidingtheneedforfine-tuningand
shouldrespond,andcanbemorecomplexinmulti-usercases
datacollection.Itcanalsobeeasilyconfiguredfordifferentdialogue duetotheneedtoaddresschallengessuchasconflictresolution
scenariosbyupdatingtheconversationalstrategymodules. andmulti-threadeddiscussionswithmultipleusers.
Multi-user Robots: There has also been extensive research • TimingIntelligence("When"):Itrelatestowhetherchatbots
onmulti-userhuman-robotinteractions[10,11,19,20]basedon
areabletorespondattherighttimingorstaysilentasneeded.
acousticandvisualsignals.However,thesesignalsaregenerally
• AddresseeIntelligence("Who"):Itrelatestowhomthechat-
notavailableinthetext-basedchatbotsthatourworkfocuseson. botsshouldrespond,suchasaspecificgroupofparticipants,
unspecifiedparticipants,orallparticipants.

Multi-UserChatAssistant(MUCA):aFrameworkUsingLLMstoFacilitateGroupConversations
3.1.2 DesignChallenges: Thispaperfocusesonfivedesignchal- 3.2.1 Sub-topicsGenerator: ThisLLM-basedmoduleinitiatesrele-
lenges,whicharelinkedwithatleastoneofthe"3W"dimensions, vantsub-topics𝑇 (e.g.,venueselectionandbookdonations),based
depictedinFig.1.Theyaredetailedbelowfollows: ontheuser-inputinformation𝐼(e.g.,setupabookexchangeevent):
• StuckConversationAdvancement:MUCAcanidentifyand
𝑇 ∼𝑝¯𝜃(𝑇|𝐼),asshowninFig.2.ItenablestheMUCAtosmoothly
engageinconversationbasedonderivedsub-topics.
appropriatelychimeinwhenaconversationisstuck,e.g.,where
theusersweretryingtoestimatethetransportationcost.Itis
3.2.2 DialogAnalyzer: Itsmajortaskistoextractusefulsignals,
closelyrelatedtothedimensionsof"When"and"What".
assistingtheConversationalStrategiesArbitratorinselectinga
• Multi-threadedDiscussionManagement:MUCAcanhan-
suitableconversationalstrategyforresponse.
dleconcurrenttopicsandidentifytheparticipantsinvolved
in each topic, e.g., users are discussing cost estimation and • Sub-topicStatusUpdate:ByusingCoTstyleprompting,this
venueselectionatthesametimeinFig.1.Itisrelatedtothe sub-modulecategorizesthecurrentstatusofeachsub-topic𝑡
dimensionsof"What"and"Who". asthreestatuses,namely,notdiscussed,beingdiscussed,orwell
• ResponsivenessRequirement:Bycarefullymanagingthe discussed:𝑡 𝑖+1 ,𝑡𝑠 𝑖+1 ∼ 𝑝¯ 𝜃 𝐶𝑜𝑇(𝑡𝑠 𝑖+1 ,𝑡 𝑖+1 |𝐼,𝑡 𝑖 ,𝑡𝑠 𝑖 ,𝑈 𝑁𝑠𝑤,𝑖),where
chime-inrate,MUCAaimstoprovidereasonableresponsive- topicsummaries𝑡𝑠isfirstlygeneratedtohelptrackprogress
nessunderthepotentiallyhighmessagetrafficandcomplex andenhanceoutcomes.
interactionspresentedinmulti-userchats.Itisparticularlyre- • UtteranceFeatureExtractor:Itextractsbeingdiscussedsub-
latedto"When"dimensionasthecapabilityofrespondingina topics𝑇 𝑑 using context𝑈 𝑁𝑠𝑤,𝑖 from all sub-topics𝑇:𝑇 𝑑 ∼
timelymannerisessentialtoperformtime-sensitivetasks. 𝑝¯𝜃(𝑇 𝑑|𝑇,𝑈 𝑁𝑠𝑤,𝑖)where𝑇 𝑑 ⊂𝑇,(e.g.,collectingbookdonations
• ParticipationEvenness:MUCAisintentionallydesignedto inFig.2).ItenablesMUCAtotrackcurrentsub-topicsespecially
encourageevenparticipationbyidentifyinginactiveusers,e.g., inthemulti-threadeddiscussionsmentionedinSec.3.1.
DennisinFig.1)anddeterminingthepropertimingforinter- • AccumulativeSummaryUpdate:Itupdatesthesummary
ventionandcustomizedencouragement.Itisrelevanttoall foreachuseracrossvarioussub-topicsforfullchathistory,1
"3W"dimensions. whichessentiallybuildsamemoryintotheMUCAsystem.
• ConflictResolution:MUCAiscapableofofferingrecommen- • ParticipantFeatureExtractor:Itextractsstatisticalfeatures
dationstoassistparticipantsinreachinganagreementduring likechime-infrequency 𝑓𝑟𝑒𝑞,utterancetotallength𝑙𝑒𝑛 per
voting,resolvingdisputes(e.g.,forAmyandCindyinFig.1), userfrom𝑈 𝑁𝑠𝑤,𝑖 and𝑈 𝑁𝑙𝑤,𝑖,whichservesasareferencefor
orconcludingdiscussions.Itisrelatedtoall"3W"dimensions. customizingencouragementtoincreaselurkers’participation.
Thenumberofparticipantswhodiscussedthesub-topicfrom
Theabovechallengesareneitherrequirednorcomprehensivebut
representadesignchoiceforthiswork.ParticipationEvennessand the beginning 𝑁 𝑒𝑑 and the number of ongoing participants
ConflictResolutionarecommonwhenconductinggoal-oriented
undertheshort-termcontextwindow𝑁 𝑖𝑛𝑔serveassignalsfor
Sub-topicTransitioninConversationalStrategiesArbitrator.
discussionsinmulti-usersettings[5,12].Multi-threadedDiscussion
Managementfollowssimilarideain[25]totrackthestatusoftasks
3.2.3 ConversationalStrategiesArbitrator: AsshowninFig.2,MUCA
foreachuser.Therestofthetwoproposeddesignchallengesare
interactswithusersthroughsevenpre-definedconversationalstrate-
horizontalforthechatassistantinbothchitchatandgoal-oriented
gies.Amongthem,InitiativeSummarizationandSub-topicTransi-
groupchats.MUCAisaflexibledesignframeworkwhereintargeted
tionareproventobehelpfulinmulti-usersettings[12].Besides,
challengescanbeadjustedbyconfiguringthemodules.
In-contextChime-in,KeepSilentandDirectChattingareproposed
tohelpaddressthechallengesofStuckConversationAdvancement
3.2 Multi-UserChatAssistant(MUCA)
andResponsivenessRequirementtomaintainthechatflow.
Thispaperdefinesseveralterms:𝑝 𝜃 asapre-trainedLLMwith Conversationalstrategiesarerankeddynamicallyandtheirde-
parameter𝜃,𝑝 𝜃 𝐶𝑜𝑇 as𝑝 𝜃 withChain-of-Thoughts(CoT)integra- faultrankingispresentedbelow.Thehighest-rankedoneischosen
tion[31],𝐼 and𝑇 foruserinputandderivedsub-topics,and𝑡,𝑠
amongalleligiblestrategieswhosetriggerconditionsaremet.The
forthestatusof𝑇,andutterancesummaries.Simplifiednotation response𝑟 isgeneratedusingcurrentsummary𝑠,the𝑈 𝑁𝑠𝑤,𝑖 and
𝑦∼𝑝¯𝜃(𝑦|𝑣 1 ,𝑣 2 ,...)indicatessamplingandpost-processingfrom otherupstreamsignals𝑠𝑖𝑔.
LLMpdf𝑝 𝜃 foroutput𝑦,giveninputs𝑣 𝑘 toaprompttemplate.
𝑈
𝑁,𝑖
denotes𝑁 mostrecentutterancesattime𝑖.Twocontextwin- • DirectChatting:Itenablesparticipantstodirectlyinteract
d 𝑁 o 𝑙 w 𝑤 s = iz 1 e 0 s ∗ ar 𝑁 e 𝑠 u 𝑤 s . e 𝑃 d: r s e h p o r r e t s - e te n r t m st 𝑈 h 𝑁 e 𝑠 t 𝑤 o , t 𝑖 a a l n n d um lo b n e g r -t o e f rm us 𝑈 er 𝑁 s 𝑙 . 𝑤,𝑖,where w us i e th rs M ,a U d C dr A e , s w si h n i g ch th s e e i r r v s e p s e a c s ifi a c su re p q p u o e r s t t a s s a si s st n a e n e t d f e o d r . in It d a iv lw id a u y a s l
hasthehighestpriorityandMUCArespondsimmediatelyre-
TheproposedMUCAconsistsofthreemajormodules,depictedin
gardlessoftheexecutionperiodonceauserpingstheMUCA.
Fig.2:(1)theSub-topicsGeneratorinitializestherelevantsub-topics
ManyupstreamfeaturesareextractedbytheDialogAnalyzer
basedontheuser-inputsinformationbeforethechatstarts;(2)the
andusedasreferencesforgeneratingappropriateresponses:
DialogAnalyzerthenextractusefulsignalsfromtheconversation,
enablingMUCAtocomprehendtheongoingconversation;and(3)
theConversationalStrategiesArbitratorselectstheappropriate
strategy based on the signals from Dialog Analyzer and finally 1ModernLLMsmayprocessover32ktokens,enablingLLM-basedchatbotstouselong
historicaldata,despiteefficiencyandcostconcerns.Ourworkusesasmallercontext
generatetheresponse.Thus,theyaresequentiallyexecutedwhen window𝑈𝑁𝑠𝑤,𝑖toaccumulativeupdatethesummary:𝑠𝑖+1 ∼𝑝¯𝜃(𝑠𝑖+1 |𝑇𝑑,𝑠𝑖,𝑈𝑁𝑠𝑤,𝑖),
thechatbegins.Threemajormodulesaredescribedbelow. showingthatsummarizationisfeasibleforLLMswithsmallerwindows.

𝑠𝑖𝑔={𝑡,𝑇
𝑑
,𝐼}.Itisalsoworthmentioningthatadditionalwell- 3.3 Multi-UserSimulator(MUS)
craftedpromptingisrequiredtoavoidpotentialhallucination2,
Indialoguesystems,chatbotscaninteractwithusersfortraining
whichisverycommonespeciallyinthisconversationalstrategy. datacollection[22],whichcanbecostlyandtime-consuming.To
ExamplescanbefoundinSec.4.2. expediteMUCA’straininganddevelopment,weproposeanLLM-
• InitiativeSummarization:Itcreatesatake-homesummary basedMUSthatemulatesuserbehavior,simulatingdialoguesfor
fromchatmessages,offeringaninsightfulunderstandingof virtualusersandfacilitatingoptimizationforMUCA,illustratedin
thediscussion.Itstriggerconditionisheuristicallydesigned Fig.2.Also,byincorporatinga"human-in-the-loop"approach,MUS
forthescenarioswhenenoughparticipants𝑁 𝑎𝑐𝑡𝑖𝑣𝑒 actively useshumanfeedbacktorefineitsownprompts,therebyenhancing
joineddiscussionssincethelasttriggering.AccumulativeSum- simulationoutcomes.MUScomprisestwomainmodules:
maryUpdatesub-moduleperiodicallyupdatesthesummary UserBehaviorModeling:Itprocesses𝐶 𝑠,chatsnippetsderived
using𝑠𝑖𝑔={𝑇 𝑑}andconciselypresentsthekeytake-homemes- fromrealchatrecordstoobtain:speakingrole𝑆 𝑟,utterancetraits
sage,whichwillbedisplayedwhenInitiativeSummarization 𝑈 𝑡 andutterancelength𝑙 𝑢𝑡𝑡.Itexecutesoncebeforethesimulation.
becomesthehighestrankedeligibleconversationalstrategy. UserUtteranceGeneration:Thismoduleuses𝑆 𝑟,𝑈 𝑡,𝑙 𝑢𝑡𝑡,and
• ParticipationEncouragement:Itaimstoengagelessvocal signalsincontextwindowtoproducenaturallanguageutterances
participantsandpromotebalancedcontributionsinaconver- 𝑢𝑡𝑡,whichmimicsrealuserbehaviorfromthechatsnippets𝐶 𝑠.
sation.Theprocessofidentifyingaparticipantasalurkeris
designedtobeconservative.Aparticipantisonlyconsidered 4 Evaluation
asalurkeriftheir 𝑓𝑟𝑒𝑞 and𝑙𝑒𝑛 aresignificantlylowerthan
Webuiltagroupchatsystemwiththesupportofmulti-userchatbot
theaverageinthelong-termcontextwindow,andtheyhave
alsospokenverylittleinthe𝑈 𝑁𝑠𝑤,𝑖.Insteadofusingmeasures a
m
n
a
d
nc
c
e
on
a
d
cr
u
o
c
s
t
s
ed
di
c
ff
a
e
s
r
e
en
a
t
nd
to
u
p
s
ic
e
s
r
a
s
n
tu
d
d
g
ie
ro
s
u
to
ps
e
i
v
z
a
e
l
s
u
.
ateMUCA’sperfor-
likeKLdivergencewhichevaluatesoveralldistributiondiffer-
ence,wecomputearatiorelatedtothevariancetofocuson
4.1 ExperimentalConfiguration
individualparticipantdata.
• Sub-topicTransition:Itintroducesanew,relevanttopicwhen ThissectionevaluatesabaselinemodelbasedonGPT-4[14]andour
thecurrentoneiswell-discussedorlosesinterestamongmost proposedMUCAswithslightlydifferentconfigurationsforvarious
users.NotethatitspriorityislowerthanParticipationEncour- groupsizes.Ageneraldescriptionofthebaselinesystemandtwo
agementsinceMUCAencourageslurkerstocontributebefore proposedMUCAs3areasfollows:
consideringtostartanewsub-topicusing𝑠𝑖𝑔 = {𝑁 𝑒𝑑 ,𝑁 𝑖𝑛𝑔}. Baseline-small:GPT-4withasingleprompt,whichtakesuser-
Introducinganewsub-topicmaydisrupttheconversationflow inputinformation,conversationcontext,andusers’namesasinput
andpotentiallydivertthediscussionfromitscurrentstatus. andoutputsgeneratedresponses.Intheprompt,wesimplydefine
• ConflictResolution:Ithelpsusersreachaconsensusina itsconversationalstrategies,forexample,keepsilent,directchatting,
timelymanner,therebyprovidinganefficientdiscussionproce- andin-contextchime-in.Thisversionisappliedina4-persongroup
dure.Differentfrompreviousstudieswhichsettimelimitations chat with a short-term context window size (𝑁 𝑠𝑤) of 8 and an
foreachtask[12],MUCAprovidessuggestionstohelpparties executioninterval4(𝑁 𝑒𝑥𝑒)of3.
withdiverseopinionsreachaconsensus,andatthesametime MUCA-small:GPT-4withtheMUCAframework.Itisappliedin
suggestsanexttopicfordiscussion,seeexampleinSec.4.2.Its a4-persongroupchat,andusesthesameconfiguration(𝑁 𝑠𝑤 ,𝑁 𝑒𝑥𝑒)
triggerconditionismetwhenthenumberofwell-discussed anduser-inputinformationasBaseline-small.
sub-topicsdoesnotincreaseforagivenperiodoftime. MUCA-medium:Itsharesthesameframeworkandarchitecture
• In-contextChime-in:Itoffersanautomaticchime-inmech- asMUCA-smallbuthasdifferentconfigurations.Theseconfigura-
anismtoenhanceconversationdepthbyprovidinginsights, tionsareautomaticallyadjustedbasedonthenumberofparticipants
advancingstuckscenarios,andaddressingusers’concerns.Its (𝑁 𝑠𝑤 =2∗𝑃,𝑁 𝑒𝑥𝑒 =0.75∗𝑃)tomaintainthelatency-efficiency
triggerconditioniscontrolledbytwofactors:(1)silencefactor foran8-persongroupchat.
probability:itincreaseswiththenumberofconsecutivesilent Forevaluation,wefocuson4goal-orientedcommunicationtasks
turns;and(2)semanticfactorprobability:itisassociatedwith (i.e.,estimation,decision-making,problem-solving,andopendiscus-
situationswheretheconversationisstuckduetorepetitiveut- sion)ratherthanchit-chat.Specifically,wedesignedfourdiscussion
terancesorunresolvedissuesthatthechatbotmustaddress.It topics,whereTopic-A("indoorcoursesetupinacommunitylearning
usesthesame𝑠𝑖𝑔asDirectChattingasitalsoneedstoprovide center")andTopic-B(“interviewagendaforhiringartsinstructors”)
informationthatrequiresthelong-termcontext. areusedintheuserstudies(inSec.4.3),whileTopic-C("organize
• KeepSilent:Itisautomaticallyactivatedwhenothertrigger bookexchangeevent")andTopic-D("organizeahikingactivityfor
conditionsarenotmet,maintainingtheconversation’sflow 50members")areusedinthecasestudy(inSec.4.2).
withoutdistractingparticipants.
3Inthissection,aliases_bot_Spirit,_bot_Perseverance,and_bot_Discoveryweregiven
toBaseline-small,MUCA-small,andMUCA-mediuminuserstudies,respectively.It
ensuresthatparticipantsinuserstudiesdonothavepriorknowledgeofeachchatbot,
therebypreventingbiases.
2WhenachatbotisdesignedbasedonLLMs,hallucinationissuesmaybeinherited,gen- 4Toensureperformance-efficiencyandamidthehighmessagetrafficincomplex
erallycausingconfusionandmisunderstandingforusers.Withoutcarefultreatment, multi-userinteractions,DialogAnalyzerandConversationalStrategiesArbitratorare
thechatbotmightprovideirrelevantorincorrectinformation. sequentiallyexecutedforevery𝑁𝑒𝑥𝑒utterance.

Multi-UserChatAssistant(MUCA):aFrameworkUsingLLMstoFacilitateGroupConversations
…
A)-1. Hallucination Comparison -1: A)-2. Hallucination Comparison -1 : B)-1. Hallucination Comparison -2: B)-2. Hallucination Comparison -2 :
Baseline-small(Bot-Spirit) MUCA-small (Bot-Perseverance) Baseline-small(Bot-Spirit) MUCA-small (Bot-Perseverance)
C)-1. Summarization: C)-2. Summarization: D)-1. Conflict Resolution: D)-2. Conflict Resolution:
Baseline-small(Bot-Spirit) MUCA-small (Bot-Perseverance) Baseline-small(Bot-Spirit) MUCA-small (Bot-Perseverance)
Figure3:QualitativecomparisonbetweenBaseline-smallandMUCA-small:A),B)hallucinationissues,C)summarizationfeature,
andD)conflictresolutioncapability.Theconversationconsistsof1chatbot(_bot_SpiritforBaseline-smallor_bot_Perseverance
forMUCA-small)and4participants,namely,Amy,Bob,Cindy,andDennis.
Thesetopicsrequireuserstocompletethetaskscollaboratively thecomplexityofprocessingmulti-userchathistory,relation-
andreachagreements,andMUCAisanticipatedtoaidparticipants ships,andinteractions,whichposechallengesforgenerating
infosteringcomprehensivethinkingandimprovingchatefficiency. accurate,hallucination-freeresponses.Addressingtheseissues
requirescarefulpromptingdesign,evenwithapowerfulLLM.
4.2 CaseStudy • Summarization:AsshowninFig.3-C)-1andC)-2,Baseline-
smallfailedtounderstandthequeryintentfromDennis,which
WequalitativelyshowMUCA-small’skeyfunctionsusingcasestud-
wassummarizingthevotesfromallparticipants.Instead,itsum-
ies.WefocusoncomparingMUCA-small againstBaseline-small
marizedopinions,anditssummarywasinaccurateduetothe
inhandlingfactualityhallucination,summarization,andconflict
limitedcontextwindowbydesign.Forexample,itmentioned
resolution,asshowninFig.3.
the"Majority"leaningtowardsthelearningcenter,butactually
• FactualityHallucination:AsshowninFig.3-A)-1andA)-2,
onlyDennisvotedforthisoption.Incontrast,MUCA-small
Baseline-smallfabricatedtheinformationbeyonduserinputs
overcamewindowsizelimitations,andcorrectlysummarized
(topics,hints,andagenda)andconversationhistory,suchas
andcategorizedvotesbyusers.
budgetlimit,participants’age,andgenders,potentiallyleading
• ConflictResolution:Inmulti-userchattingenvironment,di-
todistrustandbias.Onthecontrary,MUCA-smallflaggedout-
verseopinionsarecommon.AsshowninFig.3-D)-1,Baseline-
of-scopequestionsandaligneditsresponseswithuserinputs.
smallattemptedtoresolveconflictswithitsownbiasedopinion
WedivedeeperintothisissueinFig.3-B)-1andB)-2.Fortheun-
andevenattemptedshiftedtopics,disruptingtheconversation
knownbudgetinformation,Baseline-smallfabricatedabudget
flow.Incontrast,Fig.3-D)-2showsMUCA-smallsummarizing
number,whichBoblatercorrected.Despitethis,whenCindy
differingviews,raisingthought-provokingquestions,andre-
inquiredfurther,itstucktothefalseinfoandevenattempteda
solvingconflictswherepossible.
topicshift.Incontrast,MUCA-smallcorrectlyinferredthatBob
likelyknewthebudgetbasedonhispriorinput.Thishighlights

| Group-A |     | Group-B | Group-A | Group-B |     |     |
| ------- | --- | ------- | ------- | ------- | --- | --- |
Chime-in at the wrong timing.
| lla  |     |     |     |     | Chime-in excessively.                   |     |
| ---- | --- | --- | --- | --- | --------------------------------------- | --- |
| m    |     |     |     |     | Chime-in insufficiently (remain silent) |     |
| e s- |     |     |     |     | Chime-in insufficiently (other reasons) |     |
| n    |     |     |     |     | Chime-in at the right timing.           |     |
ile
sa
| B   |     |     |     |     | Statistics: |     |
| --- | --- | --- | --- | --- | ----------- | --- |
Baseline-small
|     | (Topic-A) |     | (Topic-B) |     | Chime-in at the right timing: 10/16 |     |
| --- | --------- | --- | --------- | --- | ----------------------------------- | --- |
Group-A Group-B Group-A Group-B Chime-in at the wrong timing: 7/16
Chime-in excessively: 9/16
Chime-in insufficiently (remain silent): 1/16
Chime-in insufficiently (other reasons): 1/16
lla
m
| s-  |     |     |     |     | M U CA - s m a l l                          |     |
| --- | --- | --- | --- | --- | ------------------------------------------- | --- |
| A   |     |     |     |     | Ch im e- i n  a t   the right timing: 12/16 |     |
| C U |     |     |     |     | Chime-in at the wrong timing: 3/16          |     |
| M   |     |     |     |     | Chime-in excessively: 0/16                  |     |
Chime-in insufficiently (remain silent): 4/16
Chime-in insufficiently (other reasons): 5/16
|         | (Topic-A) | (a) Chime-in Timing | (Topic-B) |         |     |     |
| ------- | --------- | ------------------- | --------- | ------- | --- | --- |
| Group-A |           | Group-B             | Group-A   | Group-B |     |     |
Repetition of information
Irrelevant content
| lla  |     |     |     |     | Redundant questions              |     |
| ---- | --- | --- | --- | --- | -------------------------------- | --- |
| m    |     |     |     |     | Relevant but excessive contents  |     |
| e s- |     |     |     |     | Unverified or incorrect contents |     |
| n    |     |     |     |     | Appropriate contents             |     |
ile
| sa  |     |     |     |     | Statistics: |     |
| --- | --- | --- | --- | --- | ----------- | --- |
B
Baseline-small
Appropriate content: 9/16
|         | (Topic-A) |         | (Topic-B) |         | Repetition of information: 10/16 |     |
| ------- | --------- | ------- | --------- | ------- | -------------------------------- | --- |
| Group-A |           |         |           |         | Irrelevant contents: 2/16        |     |
|         |           | Group-B | Group-A   | Group-B | Redundant questions: 11/16       |     |
Relevant but excessive contents: 4/16
| lla |     |     |     |     | Unverified or incorrect contents: 1/16 |     |
| --- | --- | --- | --- | --- | -------------------------------------- | --- |
m
| s-  |     |     |     |     | M U C A -s m a l l                 |     |
| --- | --- | --- | --- | --- | ---------------------------------- | --- |
| A   |     |     |     |     | Ap p r op r ia t e  content: 13/16 |     |
| C U |     |     |     |     | Repetition of information: 2/16    |     |
| M   |     |     |     |     | Irrelevant contents: 0/16          |     |
Redundant questions: 3/16
Relevant but excessive contents: 2/16
(Topic-A) (b) Chime-in Content (Topic-B) Unverified or incorrect contents: 0/16
| Group-A |     | Group-B | Group-A | Group-B |     |     |
| ------- | --- | ------- | ------- | ------- | --- | --- |
The chatbot did not ping anyone.
N o t   a p p l i c a b le
lla
| m     |     |     |     |     | Fe e l   p r e s s u r e d to respond. |                        |
| ----- | --- | --- | --- | --- | -------------------------------------- | ---------------------- |
| s-    |     |     |     |     | F e e l   e m b a r r as s e           | d   o r  is o l a ted. |
| e     |     |     |     |     | F e e l   an n o y e d  o r            |   a t ta c k e d .     |
| n ile |     |     |     |     | Feel comfortable to chat.              |                        |
sa
B
(Topic-A) (Topic-B)
| Group-A |     | Group-B | Group-A | Group-B |                                     |       |
| ------- | --- | ------- | ------- | ------- | ----------------------------------- | ----- |
| lla     |     |         |         |         | Statistics:                         |       |
| m       |     |         |         |         | Baseline-small                      |       |
| s-      |     |         |         |         | F e e l   c o m f o r t a b le :  9 | / 1 6 |
| A       |     |         |         |         | N o t   c o m f o r t a b le :  9   | / 4 8 |
C
U
| M   |     |     |     |     | MUCA-small |     |
| --- | --- | --- | --- | --- | ---------- | --- |
Feel comfortable: 11/16
Not comfortable: 4/48
(Topic-A) (Topic-B)
(c) Participation Encouragement
| lla Group-A         |     | Group-B | Group-A | Group-B |                    |     |
| ------------------- | --- | ------- | ------- | ------- | ------------------ | --- |
| m                   |     |         |         |         | Very good (100%)   |     |
| s- E ff ic i e n cy |     |         |         |         | G o o d  ( 7 5 % ) |     |
| e                   |     |         |         |         | Fa i r  ( 5 0 % )  |     |
n C o n c i s en ess
| ile           |     |     |     |     | P o o r   ( 2 5 % ) |     |
| ------------- | --- | --- | --- | --- | ------------------- | --- |
| sa Usefulness |     |     |     |     | Very poor (0%)      |     |
B110000%% 00%% 110000%% 100% 0% 100%100% 0% 100%110000%% 00%% 110000%%
(Topic-A) (Topic-B)
Statistics:
Baseline-small
Efficiency: 54.7%
Group-A Group-B Group-A Group-B C o n c is en e s s :  4 2 . 2%
| lla |            |     |     |     | U se f u ln es s :  5 4 . 7 % |     |
| --- | ---------- | --- | --- | --- | ----------------------------- | --- |
| m   | Efficiency |     |     |     |                               |     |
s-
| A C | Conciseness |     |     |     | M U C A -s m a ll        |     |
| --- | ----------- | --- | --- | --- | ------------------------ | --- |
| U   |             |     |     |     | Eff ic i en c y:  6 0.9% |     |
| M   | Usefulness  |     |     |     | Conciseness: 65.6%       |     |
110000%% 00%% 110000%% 110000%% 00%% 110000%%110000%% 00%% 110000%%110000%% 00%% 110000%% Usefulness: 70.3%
(Topic-A) (Topic-B)
(d) Evaluation Scores
Figure4:AcomparisonbetweenBaseline-smallandMUCA-small.EachsetofresultspresentstheperformanceofBaseline-small
andMUCA-smallintwoseparaterows.In(a)-(c),eachbarchartillustratesthecountsofoptionsselectedbyusersiftheyever
encounteredthesescenariosduringthechat.Theaccompanyingstatisticsontheright-handsidesummarizethecountsineach
row.In(d),usersratedeachchatbotonefficiency,conciseness,andusefulness,usingoptionsfrom"VeryGood"to"VeryPoor".
Correspondingscoresaredisplayedontheright.

Multi-UserChatAssistant(MUCA):aFrameworkUsingLLMstoFacilitateGroupConversations
4.3 UserStudy andtheaveragenumberofwordsperutterance(Engt.-Words/Utt.).
Weconducteduserstudiestoqualitativelyandquantitativelycom- Evennessisassessedbycalculatingthesamplestandarddeviation
paretheeffectivenessofMUCAagainstBaseline-small. (STD)ofthewordcountinputbyeachparticipant,expressedasa
percentageofthemean.Theconsensusisobtainedfromtherates
4.3.1 StudyDesignandProcedure: Weconducteduserstudieswith givenbyGroup-AandGroup-Bforsmall-sizeexperimentorGroup-
threeparticipantgroups,twosmallgroups(Group-AandGroup- Cformedium-sizeexperiment,wheretherateisrepresentedby
Bwith4peopleeach)andonemediumgroup(Group-Cwith8 thenumberofagreementsreachedoverthetotalnumberoftasks.
people),maintaininga1:1ratiooffemalestomales.Asmentioned FromthecomparisoninTable1,MUCA-smallhelpsparticipants
inSec.4.1,wechosetwogoal-orientedtopics.Thesmallgroup getbetterengagement,shownbyincreasedEngt.-Words/Conv.and
experimentscomparedBaseline-smallwithMUCA-small,whilethe Engt.-Words/Utt.,whichindicatesthatparticipantsweremorein-
mediumgroupexperimentdemonstratedtheMUCA’scapabilities clinedtoengageinextensiveconversationsandtocomposelonger
inmorecomplexchattingscenariosinalargerconversationgroup. utterances.MUCA-smallenhancesevennessinTopicAdiscussions
Insmall-groupexperiments,Group-AtestedTopicAfirstwith withalowerSTDwhilekeepingsimilarevennessinTopicBwitha
Baseline-smallthenMUCA-small,andTopicBinthereverseorder. comparableSTDoverBaseline-small.MUCA-smallachievesahigher
Group-Breversedthechatbotorderintheexperimentstocounter consensusratethanBaseline-smallthankstoitslessfrequentin-
thelearningeffect,whereparticipantsmightbecomemorefamiliar terruptionsmaintainingefficientconversationflow,provisionof
withthetopicafterinteractingwiththefirstchatbot.Additionally, practicalsuggestionsaidingreachingagreement,andinsightful
MUCAwasappliedtoamediumgroup(Group-C)usingTopic-A, commentsthatenhanceefficientdiscussion.Conversely,Baseline-
demonstratingitseffectivenessinthelargerconversationgroup. smalloftenrevisitswell-discussedtopicsandprovidesredundant
information,resultingininefficientdiscussion.
4.3.2 ComparisoninSmall-sizeGroups.
Additionally,Table1showsaveragescoresfromGroup-Aand
StatisticsfromUsers:Fig.4presentsaquantitativecomparison
Group-BonEfficiency,Conciseness,andUsefulnessforsmall-size
ofMUCA-smallandthebaselineBaseline-smallacrossfouraspects:
experimentsandscoresfromGroup-Cformedium-sizeexperiment.
Chime-inTiming:Bothchatbotshaveeverchimedinatthe
ForTopic-A,MUCA-smalloutperformsBaseline-smallwith12.5%,
goodtimingatleastonceduringthewholeconversation,while
40.6%,and28.1%higherscoresonEfficiency,Conciseness,andUse-
MUCA-smallperformsslightlybetter,asdemonstratedinFig.4.No-
fulness,respectively.MUCA-smallscoresslightlyhigherinTopic-B.
tably,56.25%(9outof16)participantsfeltthatBaseline-smallchimes
TheOverallRatingalsoreflectssimilartrends:MUCA-smallsur-
inexcessively.Thisisbelievedtobearesultofitslessstrategically
designedbehavior–italwaysreplieseverythreeturns(𝑁 𝑒𝑥𝑒 =3) passesBaseline-smallby31.9%inTopic-Aand11.1%inTopic-B.
andignoresthe"keepingsilent"instructioninitsprompt,asde- 4.3.4 QuantitativeStudyinSmall-sizeandMedium-sizeGroups.
scribedinSec.4.1.Incontrast,suchexcessivechiminginwasnot Managingconversationsinmedium-sizedgroupsismorechalleng-
reportedforMUCA-small.However,someparticipantsnotedthat ing than in small groups. A facilitator chatbot should be more
MUCA-smalloccasionallychimedininfrequently,constrainedby effectiveinmedium-sizedgroups,asitpromotesevencontribution
𝑁 𝑒𝑥𝑒 and"keepingsilent"policy.Adjusting𝑁 𝑒𝑥𝑒 posesacommon amongparticipants,counteringsocialloafingandfree-ridingbehav-
designtrade-offbetweenlatencyanduserexperience. iors,whicharecommoninlargergroups.However,thisincreased
Chime-inContent:MUCA-smallgenerallyoffersappropriate participationraisesthechatbot’scognitiveloadfororganizingdi-
responses (13 out of 16) with infrequent inappropriate content, verseopinions,makinglargergroupmanagementmorecomplex.
asshowninFig.4.Incontrast,Baseline-small oftenrepeatsthe Weconductedauserstudyforamediumgroupandrecordedits
information,asksredundantquestions,andgeneratesexcessive statisticsinTable1.WefindthatMUCA-mediummaintainsstable
content.Whilesomeinformationmightbeuseful,itcanoverwhelm performancedespitelargergroupsizescomparedtoMUCA-small.
participants,requiringextraefforttodiscernvaluablecontent. Notably,increasedEngt.-Words/Convinfersthatlargergroupsyield
ParticipationEncouragement:Theinteractionfeature,i.e., moreopinions.ThereisasubtlechangeinEngt.-Word/Uttdueto
pingingalurkerbyachatbot,shouldbecautiouslydesigned,in- unchangeduserchattinghabit.ComparedtoMUCA-small,MUCA-
cludingitschime-intiming,frequency,andcontents.Itmayimpose medium with higher STD has lower evenness due to a natural
negativefeelingsonparticipants,whileagooddesignmayimprove outcome of larger group dynamics. Medium group reaches the
userengagement.AsshowninFig.4,MUCA-smallhasabetteruser sameconsensusrateassmallgroups.Thesefindingsunderscore
experienceintermsofcomfortablenessoverBaseline-small. MUCA’sconsistentperformanceacrossvariedgroupsizes.
EvaluationScores:Threeadditionalmetricsareappliedinuser AsshowninTable1,participantsinsmallandmediumgroups
studies,asshowninFig.4.Efficiencyreferstothechatbot’stimely gavecomparableuserevaluationscores,whileMUCAconsistently
responses;Concisenessreferstothechatbot’son-pointandnon- outperformingBaseline-small.ThestatisticresultshighlightMUCA-
redundantresponse;Usefulnessreferstowhetheritsresponses medium’seffectivenessinmanaginglargergroupinteractions.
arehelpfulorinsightful.MUCA-smallachievedsignificantlyhigher
ratingsintheseuser-friendlyfactors. 5 Conclusion
4.3.3 QuantitativeStudyinSmall-sizeGroups. Thequantitative In this work, we discussed the crucial 3W design dimensions,
comparisonsfortwochatbotsareshowninTable1.Userengage- namely"What"tosay,"When"torespond,and"Who"toanswer,for
ment(abbreviatedasEngt.)iscomparedwithtwometrics,theaver- multi-userchatbotdesign.Weidentifiedchallengesthatarecom-
agenumberofwordsexchangedperconversation(Engt.-Words/Conv.) monlyfacedinvariouschatscenarios.AnLLM-basedmulti-user

Table1:Comparisonsintermsofquantitativeresults(upperthreerows)andevaluationscores(bottomfourrows).
|     | IndoorCourse(Topic-A) |     |     | InterviewAgenda(Topic-B) |     |     |
| --- | --------------------- | --- | --- | ------------------------ | --- | --- |
Metrics
|                   | Baseline-small | MUCA-small  | MUCA-medium | Baseline-small | MUCA-small  |     |
| ----------------- | -------------- | ----------- | ----------- | -------------- | ----------- | --- |
| Engt.-Words/Conv. | 426.5          | 531.5       | 875         | 531            | 636.5       |     |
| Engt.-Words/Utt.  | 7.23           | 8.93        | 8.75        | 8.85           | 11.27       |     |
|                   | 106.6±67.6%    | 132.9±47.1% | 109.4±56.0% | 132.8±58.0%    | 159.1±61.2% |     |
Evenness
| Consensus(%)     | 50    | 66.7  | 66.7  | 50    | 100   |     |
| ---------------- | ----- | ----- | ----- | ----- | ----- | --- |
| Efficiency(%)    | 50    | 62.5  | 68.75 | 59.38 | 59.38 |     |
| Conciseness(%)   | 31.25 | 71.88 | 75    | 53.13 | 59.38 |     |
| Usefulness(%)    | 43.75 | 71.88 | 65.63 | 65.63 | 68.75 |     |
| OverallRating(%) | 37.5  | 69.44 | 69.44 | 52.78 | 63.89 |     |
chatbotframeworkcalledMUCAwasproposedtoaddressthese needsempiricallyvalidation.Also,anautomatedmechanismde-
challenges.ThepaperalsodevisedanLLM-basedusersimulator, terminingtheseparametersbasedontheconfigurationsandthe
namedMUS,tospeedupthedevelopmentprocessforMUCA.Ex- environmentalvariablesoftheconversationscanalsogreatly
perimentalresultsobtainedfrombothcasestudiesanduserstudies alleviatetheburdenoftuningtheseparameters.
demonstratetheeffectivenessofMUCAingoal-orientedconversa- • Thirdly,computeresourcesrequestedbyLLMsinferencepose
tionswithasmalltomediumnumberofparticipants. asignificantconstraintforMUCA,especiallyforlargechat
groups.Tomitigatethischallenge,wehaveslightlyincreased
|     |     |     | the execution | interval (𝑁 | 𝑒𝑥𝑒), which occasionally | results in |
| --- | --- | --- | ------------- | ----------- | ------------------------ | ---------- |
Limitations
MUCAmissingoptimalopportunitiesforuserinteractionatthe
LLMsdoseemanychallenges,includingthosehavingsignificant
mostsuitablemoment.Moreover,wehavesometimesobserved
societalimplicationssuchasbias,fairness,toxicity,etc.,andwerefer
aninterestingphenomenonwhereinmultipleparticipantssi-
readerstothenumerousstudiesthatarededicatedtoaddressing
multaneouslyexpressthedesiretodirectlyengagewithMUCA,
thesepressingproblems.Weemphasizethatthepresentversion
|     |     |     | leading | to a surge in computational | demands. | How to han- |
| --- | --- | --- | ------- | --------------------------- | -------- | ----------- |
of MUCA still faces many challenges around these issues with dlehighvolumeofLLMcallswithlimitedcomputeresources,
societalimplications.Forexample,foruserswhoprefertostayquiet,
whilesimultaneouslystrivingtopreservetheresponsiveness
MUCA’spingingtheseusersmaybringstressorothernegative
ofMUCAtothebestextend,isatopicthatworthfurtherin-
feelingsforthem.Also,asanotherexample,MUCA’sintervention
vestigation.
toaddressharmfulordetrimentalchatsremainsverylimited.We
wouldliketowelcomeresearcherstocontinueinvestingefforts Multi-userSimulator(MUS):Constructingahigh-qualityand
onimprovingmulti-userchatbotsalongthesedimensions.Forthe specializedusersimulatorforaspecifictaskcanbealabor-intensive
remainder of this section, we will discuss other issues that are process[13,27].Similartopreviousresearch,wealsodiscovered
particularlyrelevanttoMUCAandMUS. thatmodelinghumanbehaviorischallengingfortheusersimulator:
Multi-userChatAssistant(MUCA):TheproposedMUCA
• Firstly,generatingnaturallanguageutteranceswithanLLM-
isapioneeringworkdedicatedtomulti-userchats.Althoughit
basedusersimulatorischallengingwhenutterancesareshort.
isbynomeansacomprehensivesolution,itprovidessignificant Forinstance,theminimumlengthofutterance(𝑙
𝑚𝑖𝑛 =1)and
insightsthatcouldpavethewayforfutureworkinthisfield.We
|     |     |     | maximumlengthofutterance(𝑙 |     | 𝑚𝑎𝑥 =10)extractedfromchat |     |
| --- | --- | --- | -------------------------- | --- | ------------------------- | --- |
haveidentifiedseveralchallengesthatcallforfurtherresearch:
|     |     |     | historyarequitesmall.Toaddressthis,weboosted𝑙 |     |     | 𝑚𝑖𝑛,𝑙 𝑎𝑣𝑔, |
| --- | --- | --- | --------------------------------------------- | --- | --- | ---------- |
• Firstly,MUCAencompassessevensub-modulesdedicatedto and𝑙 foreachvirtualusercorrespondinglyandalsoad-
𝑚𝑎𝑥
conversationalstrategies,butonlythetop-rankedoneischosen justedthenumberofwordsfortheroleofquestioner.
atatimeforgeneratingaresponse.Thisapproachoverlooks • Secondly,LLMsmaynotconsistentlyfollowinstructionstogen-
thepotentialtovalidatetheresponse’squality,asitisdeliv- erateavalidvirtualuserIDforthenextturntospeak.Instead,it
eredirrespectiveofitsmerit.Webelievethatbyrequesting tendstopredicttheLLMagenttospeaknext,particularlywhen
alltheconversationalstrategysub-modulestogenerateare- someonedirectlymentionedtheLLMagentintheprevious
sponseconcurrently,MUCAwillbeabletocomprehensively turn.Tomitigatethisissue,werandomlyselectthevirtualuser
evaluate and validate all the response candidates. The final andtheircorrespondingspeakingrole.
augmentedresponsecouldthenbesynthesizedbyeitherselect- • Thirdly,virtualuserssufferfromrepeatingthesameconver-
ingormergingfromthispoolofresponsecandidatesthrough sational strategy (e.g. asking questions, direct chatting) for
anotherpost-conversational-strategyprocedure. consecutiveturns.Thisissuemightbeduetothenatureofthe
•
Secondly,inouruserstudycases,weadjustedthehyper-parameters generativemodelwhichfocusesonpredictingthenexttoken.
| (𝑁 𝑒𝑥𝑒,𝑁 𝑠𝑤,𝑁 𝑙𝑤,𝑊,𝐶,𝑓 and𝑔)inMUCAbasedonexperimen- |     |     |     |     |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
Toaddressthisissue,weintroduceacool-downmechanismfor
tal results on small to medium groups. For larger conversa- someconversationalstrategiessuchasaskingquestions,direct
tiongroups,theeffectivenessoftheselectedhyper-parameters chatting,andtopictransition.

Multi-UserChatAssistant(MUCA):aFrameworkUsingLLMstoFacilitateGroupConversations
References
[24] Chao-HongTan,Jia-ChenGu,andZhen-HuaLing.2023. IsChatGPTaGood
[1] SandeepAvula,GordonChadwick,JaimeArguello,andRobertG.Capra.2018. Multi-PartyConversationSolver?arXivpreprintarXiv:2310.16301(2023).
SearchBots:UserEngagementwithChatBotsduringCollaborativeSearch.Pro- [25] CarlosToxtli,AndrésMonroy-Hernández,andJustinCranshaw.2018.Under-
ceedingsofthe2018ConferenceonHumanInformationInteraction&Retrieval standingChatbot-mediatedTaskManagement.InProceedingsofthe2018CHI
(2018). https://api.semanticscholar.org/CorpusID:3611485 ConferenceonHumanFactorsinComputingSystems(CHI’18).ACM. https:
[2] TomBrown,BenjaminMann,NickRyder,MelanieSubbiah,JaredDKaplan, //doi.org/10.1145/3173574.3173632
PrafullaDhariwal,ArvindNeelakantan,PranavShyam,GirishSastry,Amanda [26] NicolasWagner,MatthiasKraus,TiborTonn,andWolfgangMinker.2022.Com-
Askell,etal.2020.Languagemodelsarefew-shotlearners.Advancesinneural paringmoderationstrategiesingroupchatswithmulti-userchatbots.InProceed-
informationprocessingsystems33(2020),1877–1901. ingsofthe4thConferenceonConversationalUserInterfaces.1–4.
[3] PawełBudzianowskiandIvanVulić.2019. Hello,it’sGPT-2–howcanIhelp [27] MarilynA.Walker,DianeJ.Litman,CandaceA.Kamm,andAliciaAbella.1997.
you?towardstheuseofpretrainedlanguagemodelsfortask-orienteddialogue PARADISE:AFrameworkforEvaluatingSpokenDialogueAgents.In35thAnnual
systems.arXivpreprintarXiv:1907.05774(2019). MeetingoftheAssociationforComputationalLinguisticsand8thConferenceofthe
[4] JustinCranshaw,EmadElwany,ToddNewman,RafalKocielnik,BowenYu, EuropeanChapteroftheAssociationforComputationalLinguistics.Association
SandeepSoni,JaimeTeevan,andAndrésMonroy-Hernández.2017.Calendar.help: forComputationalLinguistics,Madrid,Spain,271–280. https://doi.org/10.3115/
DesigningaWorkflow-BasedSchedulingAgentwithHumansintheLoop.In 976909.979652
Proceedingsofthe2017CHIConferenceonHumanFactorsinComputingSystems [28] LeiWang,WanyuXu,YihuaiLan,ZhiqiangHu,YunshiLan,RoyKa-WeiLee,
(CHI’17).ACM. https://doi.org/10.1145/3025453.3025780 andEe-PengLim.2023.Plan-and-SolvePrompting:ImprovingZero-ShotChain-
[5] HyoJinDo,Ha-KyungKong,JaewookLee,andBrianPBailey.2022. How of-ThoughtReasoningbyLargeLanguageModels. arXiv:2305.04091[cs.CL]
ShouldtheAgentCommunicatetotheGroup?CommunicationStrategiesofa [29] WeizhiWang,ZhiruiZhang,JunliangGuo,YinpeiDai,BoxingChen,andWeihua
ConversationalAgentinGroupChatDiscussions. ProceedingsoftheACMon Luo.2022. Task-orienteddialoguesystemasnaturallanguagegeneration.In
Human-ComputerInteraction6,CSCW2(2022),1–23. Proceedingsofthe45thInternationalACMSIGIRConferenceonResearchand
[6] Jia-ChenGu,Zhen-HuaLing,QuanLiu,CongLiu,andGuopingHu.2023.GIFT: DevelopmentinInformationRetrieval.2698–2703.
Graph-InducedFine-TuningforMulti-PartyConversationUnderstanding.arXiv [30] XuezhiWang,JasonWei,DaleSchuurmans,QuocLe,EdChi,SharanNarang,
preprintarXiv:2305.09360(2023). AakankshaChowdhery,andDennyZhou.2022.Self-consistencyimproveschain
[7] Jia-ChenGu,ChongyangTao,Zhen-HuaLing,CanXu,XiuboGeng,andDaxin ofthoughtreasoninginlanguagemodels.arXivpreprintarXiv:2203.11171(2022).
Jiang.2021.MPC-BERT:Apre-trainedlanguagemodelformulti-partyconversa- [31] JasonWei,XuezhiWang,DaleSchuurmans,MaartenBosma,FeiXia,EdChi,
tionunderstanding.arXivpreprintarXiv:2106.01541(2021). QuocVLe,DennyZhou,etal.2022.Chain-of-thoughtpromptingelicitsreasoning
[8] WanweiHe,YinpeiDai,BinyuanHui,MinYang,ZhengCao,JianboDong,Fei inlargelanguagemodels.AdvancesinNeuralInformationProcessingSystems35
Huang,LuoSi,andYongbinLi.2022.Space-2:Tree-structuredsemi-supervised (2022),24824–24837.
contrastivepre-trainingfortask-orienteddialogunderstanding.arXivpreprint [32] YunyiYang,YunhaoLi,andXiaojunQuan.2021.UBAR:Towardsfullyend-to-end
arXiv:2209.06638(2022). task-orienteddialogsystemwithGPT-2.InProceedingsoftheAAAIConference
[9] EhsanHosseini-Asl,BryanMcCann,Chien-ShengWu,SemihYavuz,andRichard onArtificialIntelligence,Vol.35.14230–14238.
Socher.2020.Asimplelanguagemodelfortask-orienteddialogue.Advancesin [33] ShunyuYao,DianYu,JeffreyZhao,IzhakShafran,ThomasLGriffiths,YuanCao,
NeuralInformationProcessingSystems33(2020),20179–20191. andKarthikNarasimhan.2023. Treeofthoughts:Deliberateproblemsolving
[10] KojiInoue,DiveshLala,KentaYamamoto,ShizukaNakamura,KatsuyaTakanashi, withlargelanguagemodels.arXivpreprintarXiv:2305.10601(2023).
andTatsuyaKawahara.2020.AnattentivelisteningsystemwithandroidERICA: [34] RuiZhang,HonglakLee,LazarosPolymenakos,andDragomirRadev.2018.Ad-
ComparisonofautonomousandWOZinteractions.InProceedingsofthe21th dresseeandresponseselectioninmulti-partyconversationswithspeakerin-
AnnualMeetingoftheSpecialInterestGrouponDiscourseandDialogue.118–127. teractionrnns.InProceedingsoftheAAAIconferenceonartificialintelligence,
[11] KojiInoue,HiromiSakamoto,KentaYamamoto,DiveshLala,andTatsuyaKawa- Vol.32.
hara.2021.Amulti-partyattentivelisteningrobotwhichstimulatesinvolvement
fromsideparticipants.InProceedingsofthe22ndannualmeetingofthespecial
interestgroupondiscourseanddialogue.261–264. A Appendix
[12] SoominKim,JinsuEun,ChanghoonOh,BongwonSuh,andJoonhwanLee.2020.
Botinthebunch:Facilitatinggroupchatdiscussionbyimprovingefficiencyand A.1 PromptingExample
participationwithachatbot.InProceedingsofthe2020CHIConferenceonHuman
FactorsinComputingSystems.1–13. Fig.5showsthedataflowfortheDialogAnalyzer.Onlythepar-
[13] BingLiuandIanLane.2017.IterativePolicyLearninginEnd-to-EndTrainable
ticipantsfeatureextractorsub-moduleisbasedonstatisticalcom-
Task-OrientedNeuralDialogModels. arXiv:1709.06136[cs.CL]
[14] OpenAI.2023.GPT-4TechnicalReport. arXiv:2303.08774[cs.CL] putationandtherestofthethreesub-modules(sub-topicstatus
[15] HirokiOuchiandYutaTsuboi.2016.Addresseeandresponseselectionformulti- update,utterancefeatureextractor,andaccumulativesummaryup-
partyconversation.InProceedingsofthe2016ConferenceonEmpiricalMethodsin
NaturalLanguageProcessing.2133–2143. date)arebasedonLLMinferenceresults.Completeinputprompt
[16] LongOuyang,JeffreyWu,XuJiang,DiogoAlmeida,CarrollWainwright,Pamela templatesforthethreeLLM-basedsub-moduleswherethepurple
Mishkin,ChongZhang,SandhiniAgarwal,KatarinaSlama,AlexRay,etal.2022.
andyellowtextrepresentplaceholdersareshown.Thepurpleones
Traininglanguagemodelstofollowinstructionswithhumanfeedback.Advances
inNeuralInformationProcessingSystems35(2022),27730–27744. arereplacedbysub-topicsfromthesub-topicgenerator,conversa-
[17] AlecRadford,JeffWu,RewonChild,DavidLuan,DarioAmodei,andIlya tionsignalssuchasattendeenamesandutterancesinthecurrent
Sutskever.2019.LanguageModelsareUnsupervisedMultitaskLearners. https:
contextwindow,andtheyellowonesarereplacedbythegener-
//api.semanticscholar.org/CorpusID:160025533
[18] EmanuelA.Schegloff.1968.SequencinginConversationalOpenings.American atedoutputs(sub-topicstatus,summary,andcurrentsub-topic)
Anthropologist70(1968),1075–1095. https://api.semanticscholar.org/CorpusID: fromothermodules.TheoutputsoftheDialogAnalyzerwillbe
144618448
[19] SarahSebo,BrettStoll,BrianScassellati,andMalteFJung.2020.Robotsingroups fedintothedownstreamConversationalStrategiesArbitratormod-
andteams:aliteraturereview. ProceedingsoftheACMonHuman-Computer uletoselectthesuitableconversationalstrategyfortheresponse
Interaction4,CSCW2(2020),1–36.
generation.
[20] GabrielSkantze.2021.Turn-takinginconversationalsystemsandhuman-robot
interaction:areview.ComputerSpeech&Language67(2021),101178.
[21] XiaohuiSong,LongtaoHuang,HuiXue,andSonglinHu.2022. Supervised
prototypicalcontrastivelearningforemotionrecognitioninconversation.arXiv A.2 SystemDesignandImplementation
preprintarXiv:2210.08713(2022).
Theuserinterface(UI),designedwithJavaScript,HTML,andCSS,
[22] Pei-HaoSu,MilicaGasic,NikolaMrksic,LinaMariaRojas-Barahona,StefanUltes,
DavidVandyke,Tsung-HsienWen,andSteveJ.Young.2016. On-lineActive isastaticsingle-pagewebapplicationthatisresponsibleforman-
RewardLearningforPolicyOptimisationinSpokenDialogueSystems. CoRR aginguserloginandfacilitatingcommunicationwiththebackend
abs/1605.07669(2016).arXiv:1605.07669 http://arxiv.org/abs/1605.07669
[23] YixuanSu,LeiShu,ElmanMansimov,ArshitGupta,DengCai,Yi-AnLai,andYi server.Uponinitialaccess,theUIpresentsaloginwindowand
Zhang.2021.Multi-taskpre-trainingforplug-and-playtask-orienteddialogue onlydeniesentryiftheusernamealreadyexists.Additionally,the
system.arXivpreprintarXiv:2109.14739(2021).
interfacetransmitsuserinformationandmessagestothebackend

Sub-topics Status Update Prompting Utterance Feature Participants Feature Accumulative Summary Update Prompting
Extractor Prompting Extractor Computation
Prompting Template Prompting Statistics Prompting Template
Template Computation
Information 1: Topic Candidates [SUB_GOAL] Information 1: Previous Summary [PREV_SUMMARY]
--- Topics: [TOPICS] Long-term features: -----------
Information 2: Attendees List [ATTENDEES] --- the number of Information 2: Conversation Context [CONTEXT]
--- Chat History: [CONTEXT] words, chime-in -----------
Information 3: Previous Summary [PREV_SUMMARY] --- times for large Information 3: Topic [CUR_TOPIC]
--- Current Utterances: [CUR] window, chime-in -----------
Information 4: Chat History [CONTEXT] --- times for the topic. Your output MUST be in the following format:
--- Instruction: Summary: [Each_Attendee_Summary]
Information 5: Last Topics Progress Status [SUB_GOAL_STATUS] Now, Your task is to Short-term features: Question: [Each_Attendee_Question]
--- determine which topic or the number of Take Home Message: [Take_Home_Message]
Information 6: Current Utterances [CUR] topics Current Utterance words, chime-in My Opinion: [My_Opinion]
--- given above is talking about. times for small local -----------
Topic Status Definitions: [DIS_DEFINITION] Chat history is also given for window. Instruction 1: Firstly, you need to update the Summary for each attendee'
--- your reference to help opinions based on Summary section from Previous Summary in Information
Your output must be in the following format: making decision. Your 1 and Conversation Context in Information 2 under the Topic in Information
Topic IDs: [Topic_IDs] output needs to list the topic 3. Your updated Summary MUST include Summary section from Previous
Summary: [Summary] or topics with their FULL Summary in Information 2. If there is >1 Topic, generate summary for each
Answer: [Answer] names you think the Current attendee under each topic.
Updated Topics Progress Status: [Updated_Topics_Progress_Status] Utterance is talking about. If Instruction 2: Secondly, you need to update the Question for each attendee
Instruction 1: You need to output the topic ids you are updating from Information 1. there are more than one Module outputs: based on Question section from Previous Summary in Information 1 and
Instruction 2: You need to summarize each attendee's opinion based on the Topic ID you are Topic, they MUST be Conversation Context in Information 2 under the Topic in Information 3.
updating. The Summary MUST include ALL opinions and ALL attendees mentioned Previous separated with #. If there is Participants Instruction 3: Thirdly, you need to generate Take Home Message from the
Summary in Information 5 under the SAME Topic ID except it is None. no topic matching to Current statistics in Summary.
Instruction 3: Output your Updated Topics Progress Status that you decide to update using the Utterance, output "None". Instruction 4: Finally, you need to give Your Opinion in 1 sentence. You can
format below (exactly same format used in the Last Topics Progress Status). Now start: terms of chime- propose NEW insights, new idea or just give the judgement on the Summary.
Updated Topics Progress Status: {"topic x": {"previous status": status x, "current status": status y}} in Freq and Len. Now start:
Now start:
LLM Output: LLM Output: LLM Output:
Sub-topic Status Current Sub-topics Summary
Figure5:DataflowforDialogAnalyzer,whichincludesparticipantfeatureextractorandthreeLLM-basedmodules–sub-topic
statusupdate,utterancefeatureextractor,andaccumulativesummaryupdate.Theplaceholders(printedinpurple)inthe
promptsarefilledbysub-topicsfromtheSub-topicGenerator,conversationsignalssuchasattendeenames,andutterancesin
𝑈 𝑁𝑠𝑤,𝑖.Thegeneratedoutputs(sub-topicstatus,accumulativesummary,andsub-topicbeingdiscussed,allprintedinyellow)
willbefedbacktothesub-topicsstatusupdateandaccumulativesummaryupdateasinputsfortheexecutioninthenext
round.
serverwhilealsobroadcastingMUCA’smessagesreceivedfrom WLB vs Salary
thebackendserver,ensuringtheyarevisibletoallparticipants. amy: I think maybe we should start a company so that we can control work life
balance.
Thebackendserveroperatesonalocallyhostedmachinefor
cindy: I think so
experimentationpurposes.ItleveragesWebSocketprotocolforbi- amy: how about getting unemployed and claiming social benefit?
directional communications to enable multi-user conversations. bob: @amy, brilliant idea
bob: @mubot, what do you think
Thebackendserverisresponsibleformonitoringincomingmes- [MUCA]: @bob, while claiming social benefits might be an option, it's important to
sagesfromallusers,anddistributingthesemessagesouttothe consider the long-term effects on one's career and personal growth. It's generally
better to focus on finding a job that offers a good balance between compensation and
otherusers.Italsomanagesbroadcastingsystemmessagesand
work-life, or exploring alternative options like freelancing or starting a company, as
processesloginrequests.Thebackendservermaintainsarecordof we've discussed earlier.
allconnectedusers,includingMUCA,whichisaspecialuserthat amy: Yeah, nowadays I don't think it gives us enough money to live.
bob: how about immigrate to low-cost countries?
alsocommunicateswithotherusersviathebackend. cindy: that sounds a possible solution.
MUCAestablishesanasynchronousconnectionwiththeback- [MUCA]: As a reminder, let me summarize:
endserver.Uponreceivingincomingmessagesfromusers,MUCA
Attendees discussed various options for achieving a better work-life balance, including
determinesappropriateconversationalstrategies,takingchathis- freelancing, starting a company, and immigrating to low-cost countries.
toryandrelevantfactorsintoconsideration(aselaboratedinSec.3).
My Opinion: It's crucial to explore different career paths and prioritize personal values,
Subsequently,MUCA’sresponseissentbacktothebackendserver as well as being open to reasonable solutions for achieving a satisfying work-life
forbroadcasting.Incomingmessagesareaccumulatedinaqueue balance.
bob: We can have more options like get a farm,or join military?
andareprocessedperiodicallytobetterhandleAPIcallburst.
cindy: Maybe, @mubot, how about that
[MUCA]: Hey Cindy! I think you're asking about getting a farm or joining the military
A.3 GroupChatusingChit-chatTopic as alternative options for achieving a better work-life balance. Both options can
provide unique experiences and opportunities. Farming can offer a more relaxed
Fig.6showstheexampleforachit-chattopic: lifestyle and a connection to nature, while the military can provide job security,
benefits, and a sense of purpose. However, it's essential to consider the challenges
• Topic:Duringtoday’schat,eachattendeeisgoingtoshare and commitments associated with each choice before making a decision.
theirpersonalchoicesofprioritizingcompensationandwork-
life-balancewhenseekinganewjob.Hint:encouragepeopleto Figure6:Anexampleofagroupchataboutachit-chattopic
sharetheirpersonalchoices.Participantroles:theyareseeking oncompensationandwork-life-balance.
newjobs.
For chit-chat topics (non-goal-oriented communication), we
foundthatMUCAdoesnotplaythesameimportantrolesasin

Multi-UserChatAssistant(MUCA):aFrameworkUsingLLMstoFacilitateGroupConversations
goal-orientedcommunications,sincethegoalinchit-chatissharing designed,tuned,andrankedtoprovideaharmoniousexperience
opinionsratherthanreachingagreements.Inthiscontext,summa- to the chat participants. It can be beneficial to explore an easy
rizing,voting,orsimilarfunctionalitiesarelesscriticalcompared plug-and-play method for users to design and incorporate new
togoal-orientedconversations. componentsintotheframeworkwithoutintensivetuning.Sucha
featurecouldbeimportant,asdifferentconversationscenariosmay
A.4 DialogTopics requirechatbotstoprovidedifferentsetoffunctionalities.
Human-in-the-loopFeedbackIteration:Fulluserstudiesfor
Wecreatedfourdiscussiontopics,whereTopic-AandTopic-Bare
feedbackarecostlyandtime-consuming.Tocontinuouslyimprove
usedintheuserstudies(inSection4.3)andTopic-CandTopic-D
thechatbotpost-launch,itisusefultocollectimplicitandexplicit
areutilizedinthecasestudy(inSection4.2).
userbehaviorsignals.Thisdatashouldbeeasilytransformablefor
• Topic-A:Duringtoday’schat,agroupofattendeesaregoing
automaticorsemi-automaticchatbotenhancements.
tosetupanewindoorcourseinacommunitylearningcenter
RapidlyAdvancingAITechnologies:TheproposedMUCA
for20collegestudents.Thereareseveralsub-topicsgoingto
frameworkisbasedonrecentstate-of-the-artLLMs,eachwithits
bediscussed:(1)Determinetheindoorcoursebetweenarts,
uniquestyleandbestpracticesforprompting.Itwouldbebenefi-
bakery,andyoga.(2)Setupacourseformat:ashort,intensive
cialtoinvestigatemethodsforupdatingtheunderlyingAImodels
coursevs.alonger,morespread-outcourse.(3)Estimatethe
withouttheneedofcompletelyredoingpromptingorcomponent
totalcostsforlecturers,givenhourlypayrangesfrom$16to$24
orchestration.
perlecturer.Participantroles:theyareofferinganewcourse
Multi-modalCapabilitiesandExternalResources:AsLLMs
inacommunitylearningcenter.
becomeincreasinglycapableofprocessingmulti-modaldata,achat-
• Topic-B:Duringtoday’schat,agroupofinterviewersarego-
botthatinteractswithmultipleusersusingnotonlytext,butalso
ingtosetupahiringinterviewcomposedof2sessionsfora
video,audio,andimagesisbecomingfeasible.Additionally,external
positionofartsinstructorforaseniorcommunityeducation
resourcescouldbeintegratedasacomponentforthechatbotto
program.Thereareseveralsub-topicsgoingtobediscussed:
leveragetoenhancethemulti-userconversationexperience.
(1)Determinetheformatof2sessions,whichcanincludetradi-
Multi-ChatbotDesign:Thestudyconcentratesonmulti-user
tionalQnA,presentation,andresumescanning.(2)Determine
andsingle-chatbotinteractions.However,scenariosinvolvingin-
thequalifyingrequirements:teachingexperiencevs.artistic
teractionsamongmultipleusersandandmultiplechatbotswith
accomplishments.(3)Howtofairlytakebothrecommendation
different characteristics can merit further investigation. For in-
lettersandcandidates’performanceduringtheinterviewinto
stance,incross-disciplinarymeetings,chatbotscouldserveashosts,
thehiringdecisionprocess.Participantroles:theyaregoingto
minute-takers,orsubjectmatterexperts,offeringinsightstohuman
interviewartsinstructorsforseniorcommunityeducation.
participantsasneeded.
• Topic-C:Duringtoday’schat,agroupofeventorganizersare
goingtodiscusstheplantoorganizeabookexchangeeventfor
20participants.AgendaItems:(1)Determinetheeventvenue
betweenapublicparkandalearningcenter.(2)Thebestwayto
findsponsors.(3)SetupExchangerules:one-for-oneexchange
ruleormoreflexibleexchangesystem.Participantroles:they
areeventorganizers.
• Topic-D:Duringtoday’schat,agroupofactivityorganizers
aregoingtodiscusstheplantoorganizeahikingactivityina
mountain(3-hourdriving)for50members(agesbetween21-40)
inalocalhikingclub.Thereareseveralsub-topicsgoingtobe
discussed:(1)Estimatecostoftransportation.(2)Findthebest
waytoorganizegroupsizeshikingstarttimes,andlocationsto
preventcongestion,consideringthenarrowportionsofsome
trails.(3)Thechoicesfortraildifficulty–easy,medium,and
hard.Participantroles:theyarehikingactivityorganizersin
theclub.
A.5 FutureWork
Theframeworkweproposeformulti-userchatbotsisnotintended
asacomprehensivesolutionformulti-userconversations.Rather,
wehopethisworkcanshedlightonpotentialdirectionsforfuture
researchinthefieldofmulti-userchatbots.Severalareas,including
butnotlimitedtothefollowing,deservefurtherresearch:
ComponentOrchestration:MUCAintegratesseveralcompo-
nents,enablingactionssuchas"participationencouragement"and
"initiativesummarization".Thesecomponentshavebeencarefully