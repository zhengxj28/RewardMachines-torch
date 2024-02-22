from src.temporal_logic.ltl_progression import get_truth_assignments


def complete_events(main_events, redundant_events):
    """
    params
    main_events:[str] representing the main events
    redundant_events:[list] a list of single char representing events that do not matter
    return
    res_list: a list of events. Each `events` must contain the main_events

    example: complete_events('a',['m','n'])=['a','am','an','amn']
    """
    res_list = []
    for events2 in get_truth_assignments(redundant_events):
        i = 0
        while i < len(events2) and main_events > events2[i]:
            i += 1
        events = events2[:i] + main_events + events2[i:]
        res_list.append(events)
    return res_list


if __name__ == "__main__":
    res1 = complete_events('a', ['c', 'd', 'e', 'f'])
    res2 = complete_events('b', ['a', 'c', 'd'])
    # TODO: len(main_events)>=2 is not supported
    res3 = complete_events('bc', ['a', 'e'])
    print()
